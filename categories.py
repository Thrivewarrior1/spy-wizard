"""Fine-grained product category catalog + classifier + search index.

Powers the hybrid search the user spec'd:

  - Each product gets a `product_category` column populated by
    `assign_product_category()` (regex over title + handle + image
    URL). Multilingual classify patterns mean a German "Kronleuchter"
    and a French "Lustre Cristal" both land on `product_category =
    "chandelier"`.

  - A search index (`TOKEN_TO_CATEGORIES`) maps every recognised
    multilingual token AND every parent-category label to the set of
    canonical category names that should match. So:

        "chandelier"   →  {chandelier}
        "Kronleuchter" →  {chandelier}        (multilingual lookup)
        "lighting"     →  {chandelier, table-lamp, floor-lamp,
                           pendant-light, ceiling-light, wall-light,
                           candle-holder, string-lights, lamp-shade}
        "shoes"        →  {sneaker, boot, sandal, slipper, slip-on,
                           heel, loafer, oxford, orthopedic-shoe}
        "earring"      →  {earring}

  - Plural / singular normalisation is folded into the index — both
    `chandeliers` and `chandelier` resolve to `{chandelier}`. No NLP
    library required; a small lemma map handles the common cases.

ORDERING MATTERS. The classify list is iterated top-to-bottom, so
function-driven wearable gadgets (smartwatch, posture-corrector,
dog-collar) come BEFORE the apparel / jewelry blocks. Otherwise a
"smartwatch" would mis-classify as "watch" (jewelry).
"""
from __future__ import annotations

import re
from typing import Iterable

# === The catalog ==================================================
# Each entry:
#   "name"            : canonical English category slug stored in
#                       Product.product_category. Used by the search
#                       to filter the result set.
#   "parent"          : umbrella category. Searching "lighting" returns
#                       all categories whose parent == "lighting".
#   "search_tokens"   : multilingual + plural variants the user might
#                       type. Each token maps back to this canonical
#                       category in TOKEN_TO_CATEGORIES.
#   "classify_patterns": regex patterns matched against title / handle /
#                       image_url to ASSIGN this category to a product.
#                       Word boundaries used aggressively to avoid
#                       false positives (\bshoe\b not "shoehorn").
PRODUCT_CATEGORIES: list[dict] = [
    # =================================================================
    # WEARABLE GADGETS — must come BEFORE jewelry / accessories so a
    # smartwatch never resolves to category=watch (jewelry).
    # =================================================================
    {
        "name": "smartwatch",
        "parent": "wearable-gadget",
        "search_tokens": ["smartwatch", "smart-watch", "smart watch", "fitness watch"],
        "classify_patterns": [
            r"\bsmart[\s\-]?watch\w*", r"\bsmartwatch\w*",
            r"\bsenior[\s\-]?smart[\s\-]?watch\w*",
            r"\bhealth[\s\-]?(?:smart[\s\-]?)?watch",
            r"\b(?:blood[\s\-]?(?:sugar|pressure|glucose)|ecg|heart[\s\-]?rate)[\s\-]?(?:smart[\s\-]?)?watch",
        ],
    },
    {
        "name": "fitness-tracker",
        "parent": "wearable-gadget",
        "search_tokens": ["fitness-tracker", "fitness tracker", "activity-tracker", "smart-band", "smart band", "smart-ring", "smart ring", "smart-glasses"],
        "classify_patterns": [
            r"\bfitness[\s\-]?(?:tracker|band)\w*",
            r"\bactivity[\s\-]?tracker\w*",
            r"\bsmart[\s\-]?(?:ring|band|glasses)\b",
        ],
    },
    {
        "name": "posture-corrector",
        "parent": "wearable-gadget",
        "search_tokens": ["posture-corrector", "posture corrector", "posture brace", "haltungskorrektur"],
        "classify_patterns": [
            r"\bposture[\s\-]?correct(?:or|er)\w*",
            r"\bposture[\s\-]?brace\w*",
            r"\bhaltungs(?:korrektur|trainer|bandage|bra|gurt)\w*",
        ],
    },
    {
        "name": "support-brace",
        "parent": "wearable-gadget",
        "search_tokens": ["brace", "support-band", "support band", "knee-brace", "back-brace", "stützband"],
        "classify_patterns": [
            r"\bsupport[\s\-]?(?:band|belt|brace)\w*",
            r"\b(?:knee|back|elbow|wrist|ankle)[\s\-]?brace\w*",
            r"\bst(?:ü|u|ue)tzg(?:ü|u|ue)rtel\w*",
            r"\bst(?:ü|u|ue)tzband\w*",
            r"\bknieorthese\w*", r"\borthese\w*",
            r"\bsciatica[\s\-]?(?:relief|brace|belt|support)\w*",
            r"\bhip[\s\-]?and[\s\-]?thigh[\s\-]?support\w*",
        ],
    },
    {
        "name": "orthopedic-insole",
        "parent": "wearable-gadget",
        "search_tokens": ["insole", "insoles", "orthopedic insole", "einlagen", "toe-spacer", "toe spacer"],
        "classify_patterns": [
            r"\borthopedic[\s\-]?insoles?\b",
            r"\borthop(?:ä|a|ae)dische?[\s\-]?einlagen\w*",
            r"\b(?:gel|memory)[\s\-]?insoles?\b",
            r"\binsoles?\b", r"\beinlagen\b",
            r"\btoe[\s\-]?spacers?\b", r"\bzehenspreizer\w*",
            r"\bbunion[\s\-]?correct(?:or|er)\w*",
            r"\bfoot[\s\-]?pain[\s\-]?(?:pad|relief|cushion|insert)\w*",
            r"\bmetatarsal[\s\-]?support\w*",
            r"\bgel[\s\-]?(?:cushion|pad|insole|insert)\w*",
        ],
    },
    {
        "name": "compression-hosiery",
        "parent": "wearable-gadget",
        "search_tokens": ["compression-sleeve", "compression sleeve", "compression-stocking", "compression stocking", "kompressionsstrümpfe"],
        "classify_patterns": [
            r"\bcompression[\s\-]?(?:sleeve|sock|stocking|legging|garment|wear|shirt)\w*",
            r"\bkompressions(?:[äa]rmel|str(?:ü|u|ue)mpfe|socke\w*|leggings?)",
        ],
    },

    # =================================================================
    # HOSE HARDWARE — must come before APPAREL.pants so 'Garden Hose'
    # doesn't classify as German pants. The English noun is utility,
    # not clothing.
    # =================================================================
    {
        "name": "hose-hardware",
        "parent": "outdoors",
        "search_tokens": ["garden hose", "garden-hose", "fire hose", "hose attachment", "expandable hose"],
        "classify_patterns": [
            r"\bgarden[\s\-]?hoses?\b",
            r"\bgarten[\s\-]?schlauch\w*",
            r"\bfire[\s\-]?hoses?\b",
            r"\bexpandable[\s\-]?hoses?\b",
            r"\bhose[\s\-]?(?:attachment|reel|nozzle|sprayer?|connector|fitting)s?\b",
            r"\b(?:high[\s\-]?pressure|water|spray)[\s\-]?hoses?\b",
        ],
    },

    # =================================================================
    # PET PROTECTIVE — also before apparel/jewelry (e.g. dog-collar
    # before necklace/jewelry collars).
    # =================================================================
    {
        "name": "dog-raincoat",
        "parent": "pet-protective",
        "search_tokens": ["dog-raincoat", "dog raincoat", "pet raincoat", "hundemantel"],
        "classify_patterns": [
            r"\bdog[\s\-]?raincoat\w*",
            r"\bpet[\s\-]?raincoat\w*",
            r"\bhundemantel\w*",
            r"\bhunderegen\w*",
        ],
    },
    {
        "name": "dog-harness",
        "parent": "pet-protective",
        "search_tokens": ["dog-harness", "dog harness", "pet harness", "hundegeschirr"],
        "classify_patterns": [
            r"\bdog[\s\-]?harness\w*",
            r"\bpet[\s\-]?harness\w*",
            r"\bhundegeschirr\w*",
        ],
    },
    {
        "name": "dog-collar",
        "parent": "pet-protective",
        "search_tokens": ["dog-collar", "dog collar", "pet collar", "led-collar", "led collar"],
        "classify_patterns": [
            r"\bled[\s\-]?(?:dog|pet|safety)[\s\-]?collars?\b",
            r"\bled[\s\-]?(?:hunde)?halsband\w*",
            r"\b(?:training|e)[\s\-]?collar\w*",
            r"\bdog[\s\-]?collar\w*",
        ],
    },
    {
        "name": "dog-muzzle",
        "parent": "pet-protective",
        "search_tokens": ["dog-muzzle", "dog muzzle", "muzzle", "maulkorb"],
        "classify_patterns": [
            r"\bdog[\s\-]?muzzle\w*",
            r"\bpet[\s\-]?muzzle\w*",
            r"\bmaulkorb\w*",
        ],
    },
    {
        "name": "dog-leash",
        "parent": "pet-protective",
        "search_tokens": ["dog-leash", "dog leash", "leash", "hundeleine"],
        "classify_patterns": [
            r"\bdog[\s\-]?leash\w*",
            r"\bpet[\s\-]?leash\w*",
            r"\bhundeleine\w*",
        ],
    },

    # =================================================================
    # LIGHTING — after wearable-gadget / pet-protective so nothing
    # else captures these first.
    # =================================================================
    {
        "name": "chandelier",
        "parent": "lighting",
        "search_tokens": [
            "chandelier", "chandeliers", "kronleuchter", "lustre",
            "lampadario", "araña", "arana", "kroonluchter",
        ],
        "classify_patterns": [
            r"\bchandeliers?\b",
            r"\bkronleuchter\w*",
            r"\blustres?\b",
            r"\blampadari(?:o)?\b",   # 'lampadari' or 'lampadario'
            r"\bara(?:ñ|n)a\b",
            r"\bkroonluchter\w*",
        ],
    },
    {
        "name": "table-lamp",
        "parent": "lighting",
        "search_tokens": [
            "table-lamp", "table lamp", "desk lamp", "bedside lamp",
            "tischlamp", "tischleucht", "lampe de table",
            "lampada da tavolo", "lámpara de mesa", "tafellamp",
        ],
        "classify_patterns": [
            r"\btable[\s\-]?lamp\w*",
            r"\bdesk[\s\-]?lamp\w*",
            r"\bbedside[\s\-]?lamp\w*",
            r"\btischlamp\w*", r"\btischleucht\w*",
            r"\blampe\s+de\s+table",
            r"\blampada\s+da\s+tavolo",
            r"\bl[áa]mpara\s+de\s+mesa",
            r"\btafellamp\w*",
        ],
    },
    {
        "name": "floor-lamp",
        "parent": "lighting",
        "search_tokens": [
            "floor-lamp", "floor lamp", "stehlamp", "stehleucht",
            "lampe de sol", "lampada da terra", "lámpara de pie", "vloerlamp",
        ],
        "classify_patterns": [
            r"\bfloor[\s\-]?lamp\w*",
            r"\bstehlamp\w*", r"\bstehleucht\w*",
            r"\blampe\s+de\s+sol",
            r"\blampada\s+da\s+terra",
            r"\bl[áa]mpara\s+de\s+pie",
            r"\bvloerlamp\w*",
        ],
    },
    {
        "name": "pendant-light",
        "parent": "lighting",
        "search_tokens": [
            "pendant-light", "pendant light", "pendant lamp",
            "hängelampe", "hangelampe", "pendelleuchte",
            "suspension", "lampada a sospensione",
        ],
        "classify_patterns": [
            r"\bpendant[\s\-]?(?:light|lamp)\w*",
            r"\bh(?:ä|a|ae)ngelamp\w*", r"\bh(?:ä|a|ae)ngeleucht\w*",
            r"\bpendelleucht\w*", r"\bpendellampe\w*",
            r"\bsuspension\b",
            r"\blampada\s+a\s+sospensione",
        ],
    },
    {
        "name": "ceiling-light",
        "parent": "lighting",
        "search_tokens": [
            "ceiling-light", "ceiling light", "ceiling lamp", "ceiling fan",
            "deckenlamp", "deckenleucht", "plafonnier", "plafondlamp",
        ],
        "classify_patterns": [
            r"\bceiling[\s\-]?(?:light|lamp|fan|fixture)\w*",
            r"\bdeckenlamp\w*", r"\bdeckenleucht\w*",
            r"\bplafonnier\w*", r"\bplafondlamp\w*",
        ],
    },
    {
        "name": "wall-light",
        "parent": "lighting",
        "search_tokens": [
            "wall-light", "wall light", "wall lamp", "sconce", "sconces",
            "wall sconce", "wandlamp", "wandleucht", "applique",
        ],
        "classify_patterns": [
            r"\bwall[\s\-]?(?:light|lamp|sconce|fixture)\w*",
            r"\bsconces?\b",
            r"\bwandlamp\w*", r"\bwandleucht\w*",
            r"\bappliques?\b",
        ],
    },
    {
        "name": "candle-holder",
        "parent": "lighting",
        "search_tokens": [
            "candle-holder", "candle holder", "candleholder", "candelabra",
            "kerzenhalter", "kandelaar", "bougeoir", "candelabro",
        ],
        "classify_patterns": [
            r"\bcandle[\s\-]?holders?\b",
            r"\bcandleholders?\b",
            r"\bcandelabra\w*",
            r"\bkerzenhalter\w*", r"\bkerzenst(?:ä|a|ae)nder\w*",
            r"\bkandelaars?\b", r"\bbougeoirs?\b",
            r"\bcandelabro\w*",
        ],
    },
    {
        "name": "string-lights",
        "parent": "lighting",
        "search_tokens": [
            "string-lights", "string lights", "fairy-lights", "fairy lights",
            "christmas lights", "lichterkette", "guirlande lumineuse",
        ],
        "classify_patterns": [
            r"\b(?:string|fairy|tape|christmas|rope)\s+lights?\b",
            r"\b(?:solar|garden|outdoor|landscape|patio|pathway|driveway)[\s\-]+(?:\w+[\s\-]+){0,2}lights?\b",
            r"\blichterkett\w*",
            r"\bguirlandes?\s+lumineuses?\b",
        ],
    },
    {
        "name": "lamp-shade",
        "parent": "lighting",
        "search_tokens": ["lamp-shade", "lamp shade", "lampshade", "lampenschirm", "abat-jour"],
        "classify_patterns": [
            r"\blamp[\s\-]?shades?\b",
            r"\blampshades?\b",
            r"\blampenschirm\w*",
            r"\babat[\s\-]?jours?\b",
            r"\bparalume\w*",
        ],
    },
    {
        "name": "light-bulb",
        "parent": "lighting",
        "search_tokens": ["light-bulb", "light bulb", "led-bulb", "led bulb", "bulb"],
        "classify_patterns": [
            r"\blight\s+bulbs?\b",
            r"\bled\s+(?:bulb|strip|panel|tube|lamp)s?\b",
        ],
    },

    # =================================================================
    # EYEWEAR ACCESSORIES — function, not fashion. Before "eyewear".
    # =================================================================
    {
        "name": "lens-cleaner",
        "parent": "eyewear-accessory",
        "search_tokens": ["lens-cleaner", "lens cleaner", "microfiber cloth", "lens-wipe", "anti-fog spray", "brillenreiniger"],
        "classify_patterns": [
            r"\blens[\s\-]?(?:cleaner|cleaning|cloth|wipe|wipes|spray|fog|case|kit|holder|polish)\w*",
            r"\bmicrofib(?:er|re)[\s\-]?(?:lens[\s\-]?)?cloth\w*",
            r"\banti[\s\-]?fog[\s\-]?spray\w*",
            r"\bbrillen(?:reiniger|tuch|t(?:ü|u|ue)cher|spray)\w*",
        ],
    },
    {
        "name": "contact-lens-supply",
        "parent": "eyewear-accessory",
        "search_tokens": ["contact-lens", "contact lens solution", "contact lens case", "eye-drops", "eye drops"],
        "classify_patterns": [
            r"\bcontact[\s\-]?lens(?:[\s\-]?(?:solution|case|cleaner|holder))?\w*",
            r"\beye[\s\-]?drops?\b",
            r"\baugentropfen\w*",
        ],
    },

    # =================================================================
    # EYEWEAR (style)
    # =================================================================
    {
        "name": "sunglasses",
        "parent": "eyewear",
        "search_tokens": [
            "sunglasses", "sonnenbrille", "lunettes de soleil",
            "gafas de sol", "occhiali da sole",
        ],
        "classify_patterns": [
            r"\bsunglasses\b",
            r"\bsonnenbrille\w*",
            r"\blunettes\s+de\s+soleil",
            r"\bgafas\s+de\s+sol",
            r"\bocchiali\s+da\s+sole",
        ],
    },
    {
        "name": "glasses",
        "parent": "eyewear",
        "search_tokens": [
            "glasses", "reading-glasses", "reading glasses",
            "progressive-glasses", "progressive glasses", "frames",
            "brille", "brillen", "lesebrille", "lunettes", "gafas", "occhiali", "bril",
        ],
        "classify_patterns": [
            r"\b(?:reading|progressive|prescription|bifocal|varifocal)[\s\-]?(?:glasses|lenses?)?\b",
            r"\bglasses\b", r"\beyewear\b", r"\bframes\b",
            r"\bbrillen?\b", r"\blesebrille\w*",
            r"\blunettes\b", r"\bgafas\b", r"\bocchiali\b", r"\bbril\b",
        ],
    },

    # =================================================================
    # FOOTWEAR
    # =================================================================
    {
        "name": "orthopedic-shoe",
        "parent": "footwear",
        "search_tokens": ["orthopedic-shoe", "orthopedic shoes", "ortho shoe", "orthoschuh"],
        "classify_patterns": [
            # Allow up to 2 words between 'orthopedic' and the noun so
            # "Orthopedic Walking Shoes" / "Orthopedic Comfort Boots"
            # both classify here. The bare 'orthopedic' adjective gets
            # captured even without a directly-following noun via the
            # German 'orthop[äa]disch' pattern below.
            r"\borthopedic[\s\-]+(?:\w+[\s\-]+){0,2}(?:shoes?|sneakers?|boots?|sandals?|slip[\s\-]?ons?|loafers?|heels?|flats?|insoles?|toe)\w*",
            r"\borthoschuh\w*",
            r"\borthop(?:ä|a|ae)disch[\s\-]+(?:\w+[\s\-]+){0,2}(?:schuhe?|sandalen?|sneakers?|stiefel)\w*",
        ],
    },
    {
        "name": "sneaker",
        "parent": "footwear",
        "search_tokens": ["sneaker", "sneakers", "trainer", "trainers", "turnschuh", "basket", "baskets", "zapatilla"],
        "classify_patterns": [
            r"\bsneakers?\b",
            r"\bturnschuh\w*",
            r"\bbaskets?\b",
            r"\bzapatillas?\b",
        ],
    },
    {
        "name": "boot",
        "parent": "footwear",
        "search_tokens": ["boot", "boots", "stiefel", "bottes", "bota", "stivali", "laarzen"],
        "classify_patterns": [
            r"\bboots?\b",
            r"\bstiefel\w*",
            r"\bbottes?\b",
            r"\bbotas?\b",
            r"\bstivali\b",
            r"\blaarzen\b",
        ],
    },
    {
        "name": "sandal",
        "parent": "footwear",
        "search_tokens": ["sandal", "sandals", "sandalen", "sandales", "sandalia", "sandali"],
        "classify_patterns": [
            r"\bsandals?\b",
            r"\bsandalen\b",
            r"\bsandales?\b",
            r"\bsandalias?\b",
            r"\bsandali\b",
        ],
    },
    {
        "name": "slipper",
        "parent": "footwear",
        "search_tokens": ["slipper", "slippers", "hausschuh", "pantoufle"],
        "classify_patterns": [
            r"\bslippers?\b",
            r"\bhausschuh\w*",
            r"\bpantoufles?\b",
        ],
    },
    {
        "name": "slip-on",
        "parent": "footwear",
        "search_tokens": ["slip-on", "slip ons", "slipon", "slip ons", "slip-ons"],
        "classify_patterns": [
            r"\bslip[\s\-]?ons?\b",
        ],
    },
    {
        "name": "heel",
        "parent": "footwear",
        "search_tokens": ["heel", "heels", "stiletto", "stilettos", "escarpins", "high-heel"],
        "classify_patterns": [
            r"\bhigh[\s\-]?heels?\b",
            r"\bheels?\b",
            r"\bstilettos?\b",
            r"\bescarpins?\b",
        ],
    },
    {
        "name": "loafer",
        "parent": "footwear",
        "search_tokens": ["loafer", "loafers", "mokassin", "mocassin"],
        "classify_patterns": [
            r"\bloafers?\b",
            r"\bmokassins?\b",
            r"\bmocassins?\b",
        ],
    },
    {
        "name": "oxford",
        "parent": "footwear",
        "search_tokens": ["oxford", "oxfords", "derby", "derby shoe"],
        "classify_patterns": [
            r"\boxfords?\b",
            r"\bderby[\s\-]?shoes?\b",
        ],
    },

    # =================================================================
    # APPAREL
    # =================================================================
    {
        "name": "dress",
        "parent": "apparel",
        "search_tokens": [
            "dress", "dresses", "kleid", "kleider", "robe", "robes",
            "vestido", "vestidos", "vestito", "vestiti", "jurk",
        ],
        "classify_patterns": [
            r"\bdress(?:es)?\b",
            # German 'kleid' is a compound suffix in titles like
            # 'Sommerkleid' / 'Brautkleid' — no leading word boundary
            # so it matches inside the compound.
            r"kleid\w*",
            r"\brobes?\b",
            r"\bvestidos?\b",
            r"\bvestit[io]\b",         # vestiti or vestito
            r"\bjurk(?:en)?\b",        # jurk or jurken
        ],
    },
    {
        "name": "skirt",
        "parent": "apparel",
        "search_tokens": ["skirt", "skirts", "rock", "röcke", "jupe", "falda", "gonna", "rok"],
        "classify_patterns": [
            r"\bskirts?\b",
            r"\br(?:ö|o|oe)ck\b",
            r"\br(?:ö|o|oe)cke\b",
            r"\bjupes?\b",
            r"\bfaldas?\b",
            r"\bgonne?\b",
            r"\brokken?\b",
        ],
    },
    {
        "name": "t-shirt",
        "parent": "apparel",
        "search_tokens": ["t-shirt", "t shirt", "tshirt", "tee", "t-shirts", "tees"],
        "classify_patterns": [
            r"\bt[\s\-]?shirts?\b",
            r"\btees?\b",
        ],
    },
    {
        "name": "shirt",
        "parent": "apparel",
        "search_tokens": ["shirt", "shirts", "hemd", "hemden", "chemise", "camisa", "camicia"],
        "classify_patterns": [
            r"\bshirts?\b",
            r"\bhemd(?:en)?\b",         # hemd or hemden (NOT hemde)
            r"\bchemises?\b",
            r"\bcamisas?\b",
            r"\bcamicia\b", r"\bcamicie\b",
        ],
    },
    {
        "name": "blouse",
        "parent": "apparel",
        "search_tokens": ["blouse", "blouses", "bluse", "blusen", "chemisier", "blusa", "camicetta"],
        "classify_patterns": [
            r"\bblouses?\b",
            r"\bbluse(?:n)?\b",         # bluse or blusen
            r"\bchemisiers?\b",
            r"\bblusas?\b",
            r"\bcamicette\b",
        ],
    },
    # Hoodie BEFORE sweater so "Hoodie Pullover Schwarz" classifies
    # as hoodie (the user-typed primary noun) instead of sweater (which
    # would also match via the 'pullover' German loanword).
    {
        "name": "hoodie",
        "parent": "apparel",
        "search_tokens": ["hoodie", "hoodies", "kapuzenpullover", "sweat à capuche"],
        "classify_patterns": [
            r"\bhoodies?\b",
            r"\bkapuzenpulli\w*",
            r"\bkapuzenpullover\w*",
            r"\bsweat\s+(?:à|a)\s+capuche",
        ],
    },
    {
        "name": "sweater",
        "parent": "apparel",
        "search_tokens": ["sweater", "sweaters", "pullover", "suéter", "maglione", "trui"],
        "classify_patterns": [
            r"\bsweaters?\b",
            r"\bpullover\w*",
            r"\bsu[ée]teres?\b",
            r"\bmaglion[ei]\b",
            r"\btrui(?:en)?\b",         # trui or truien
        ],
    },
    {
        "name": "cardigan",
        "parent": "apparel",
        "search_tokens": ["cardigan", "cardigans", "strickjacke"],
        "classify_patterns": [
            r"\bcardigans?\b",
            r"\bstrickjacke\w*",
        ],
    },
    {
        "name": "jacket",
        "parent": "apparel",
        "search_tokens": ["jacket", "jackets", "jacke", "jacken", "veste", "chaqueta", "giacca", "jas"],
        "classify_patterns": [
            r"\bjackets?\b",
            r"\bjacke(?:n)?\b",         # jacke or jacken
            r"\bvestes?\b",
            r"\bchaquetas?\b",
            r"\bgiacche?\b",
            r"\bjas\b", r"\bjassen\b",
        ],
    },
    # Bathrobe BEFORE coat — German 'Bademantel' contains 'mantel' so
    # the no-boundary coat pattern would otherwise claim it.
    {
        "name": "bathrobe",
        "parent": "apparel",
        "search_tokens": ["bathrobe", "bathrobes", "bademantel", "peignoir", "badjas", "robe-de-chambre"],
        "classify_patterns": [
            r"\bbathrobes?\b", r"\bbathoobe\b",
            r"\bbademantel\w*", r"\bbademaentel\w*",
            r"\bpeignoirs?\b",
            r"\bbadjassen?\b", r"\bbadjas\b",
        ],
    },
    {
        "name": "coat",
        "parent": "apparel",
        "search_tokens": ["coat", "coats", "mantel", "mäntel", "manteau", "abrigo", "cappotto"],
        "classify_patterns": [
            r"\bcoats?\b",
            r"m(?:ä|a|ae)ntel\w*", r"mantel\w*",   # German compound — no leading \b
            r"\bmanteau\w*", r"\bmanteaux\b",
            r"\babrigos?\b",
            r"\bcappott[io]\b",         # cappotti or cappotto
        ],
    },
    {
        "name": "pants",
        "parent": "apparel",
        "search_tokens": ["pants", "trousers", "hose", "hosen", "pantalon", "pantaloni", "broek"],
        "classify_patterns": [
            r"\bpants\b", r"\btrousers\b",
            r"\bhose\b", r"\bhosen\w*",
            r"\bpantalons?\b", r"\bpantaloni\b",
            r"\bbroeken?\b",
        ],
    },
    {
        "name": "jeans",
        "parent": "apparel",
        "search_tokens": ["jeans", "jean", "denim"],
        "classify_patterns": [
            r"\bjeans\b", r"\bdenim\b",
        ],
    },
    {
        "name": "shorts",
        "parent": "apparel",
        "search_tokens": ["shorts", "short"],
        "classify_patterns": [
            r"\bshorts\b",
        ],
    },
    {
        "name": "jumpsuit",
        "parent": "apparel",
        "search_tokens": ["jumpsuit", "jumpsuits", "overall", "overalls"],
        "classify_patterns": [
            r"\bjumpsuits?\b",
            r"\boveralls?\b",
        ],
    },
    # bathrobe lives above the coat block so 'Bademantel' wins
    # precedence over 'mantel'. The duplicate that used to live here
    # was removed; pajamas follows the bathrobe block now.
    {
        "name": "pajamas",
        "parent": "apparel",
        "search_tokens": ["pajamas", "pyjamas", "schlafanzug", "pijama"],
        "classify_patterns": [
            r"\bpajamas?\b", r"\bpyjamas?\b",
            r"\bschlafanzug\w*", r"\bschlafanz(?:ü|u|ue)g\w*",
            r"\bnachthemd\w*",
            r"\bpijamas?\b",
        ],
    },
    {
        "name": "swimwear",
        "parent": "apparel",
        "search_tokens": ["swimsuit", "swimsuits", "bikini", "swimwear", "bademode", "maillot", "bañador"],
        "classify_patterns": [
            r"\bswimsuits?\b",
            r"\bbikinis?\b",
            r"\bswimwear\b",
            r"\bbademode\w*",
            r"\bbadeanz(?:ü|u|ue)g\w*",
            r"\bmaillots?\b",
            r"\bba(?:ñ|n)ador\w*",
        ],
    },

    # =================================================================
    # INTIMATES
    # =================================================================
    {
        "name": "bra",
        "parent": "intimates",
        "search_tokens": ["bra", "bras", "bh", "soutien-gorge", "soutien gorge", "reggiseno", "sostén", "beha"],
        "classify_patterns": [
            r"\bbras?\b",
            r"\bbh\b", r"\bb\.?h\.?\b",
            r"\bsoutien[\s\-]?gorges?\b",
            r"\breggisen[oi]\b",
            r"\bsost[ée]nes?\b", r"\bsost[ée]n\b",
            r"\bbehas?\b",
        ],
    },
    # Boxer BEFORE panties so "Boxer Briefs Men" classifies as boxer
    # (the lead noun) instead of panties via the shared 'briefs?'
    # alternation.
    {
        "name": "boxer",
        "parent": "intimates",
        "search_tokens": ["boxer", "boxers", "boxer-briefs", "boxer briefs", "calzoncillo"],
        "classify_patterns": [
            r"\bboxers?\b",
            r"\bboxer[\s\-]?briefs?\b",
        ],
    },
    {
        "name": "panties",
        "parent": "intimates",
        "search_tokens": ["panty", "panties", "briefs", "underpants", "culotte", "bragas", "unterhose"],
        "classify_patterns": [
            r"\bpanty\b", r"\bpanties\b",
            r"\bbriefs?\b", r"\bunderpants\b",
            r"\bculottes?\b",
            r"\bbragas?\b", r"\bbraguitas?\b",
            r"\bunterhose\w*",
        ],
    },
    {
        "name": "thong",
        "parent": "intimates",
        "search_tokens": ["thong", "thongs"],
        "classify_patterns": [
            r"\bthongs?\b",
        ],
    },
    {
        "name": "lingerie",
        "parent": "intimates",
        "search_tokens": ["lingerie", "dessous"],
        "classify_patterns": [
            r"\blingerie\b",
            r"\bdessous\b",
        ],
    },
    {
        "name": "shapewear",
        "parent": "intimates",
        "search_tokens": ["shapewear", "bodyshaper", "body-shaper", "waist-trainer", "waist trainer"],
        "classify_patterns": [
            r"\bshapewear\b",
            r"\bbody[\s\-]?shapers?\b",
            r"\bwaist[\s\-]?trainers?\b",
        ],
    },
    {
        "name": "underwear-generic",
        "parent": "intimates",
        "search_tokens": ["underwear", "intimo", "ondergoed", "unterwäsche", "ropa interior", "sous-vêtements"],
        "classify_patterns": [
            r"\bunderwear\b",
            r"\bunterw(?:ä|a|ae)sche\w*",
            r"\bsous[\s\-]?v[êe]tements?\b",
            r"\bropa[\s\-]?interior\b",
            r"\bintimo\b",
            r"\bbiancheria[\s\-]?intima\b",
            r"\bondergoed\b",
            r"\bnahtlose?\b", r"\bseamless\b",
        ],
    },
    {
        "name": "hosiery",
        "parent": "intimates",
        "search_tokens": ["tights", "stockings", "hosiery", "strumpfhose", "collants", "calze", "kousen"],
        "classify_patterns": [
            r"\bhosiery\b", r"\btights\b", r"\bstockings?\b",
            r"\bstr(?:ü|u|ue)mpfh(?:ö|o|oe)se\w*",
            r"\bcollants\b",
            r"\bmedias\b",
            r"\bcalze\b",
            r"\bkousen\b",
        ],
    },
    {
        "name": "socks",
        "parent": "intimates",
        "search_tokens": ["sock", "socks", "socken", "chaussettes", "calcetines", "sokken"],
        "classify_patterns": [
            r"\bsocks?\b",
            r"\bsocken\b",
            r"\bchaussettes\b",
            r"\bcalcetines\b",
            r"\bsokken\b",
        ],
    },

    # =================================================================
    # JEWELRY
    # =================================================================
    {
        "name": "necklace",
        "parent": "jewelry",
        "search_tokens": ["necklace", "necklaces", "halskette", "kette", "collier", "collana", "ketting"],
        "classify_patterns": [
            r"\bnecklaces?\b",
            r"\bhalskette\w*", r"\bkette\b", r"\bketten\b",
            r"\bcollier\w*",
            r"\bcollana\b", r"\bcollane\b",
            r"\bketting\b",
        ],
    },
    {
        "name": "earring",
        "parent": "jewelry",
        "search_tokens": ["earring", "earrings", "stud", "studs", "hoop", "hoops", "ohrring", "boucle d'oreille", "pendiente", "orecchino"],
        "classify_patterns": [
            r"\bearrings?\b",
            r"\bohrring\w*",
            r"\bboucles?[\s\-]?d'?oreilles?\b",
            r"\bpendientes?\b",
            r"\borecchin[oi]\b",
            r"\boorbel\b",
        ],
    },
    {
        "name": "bracelet",
        "parent": "jewelry",
        "search_tokens": ["bracelet", "bracelets", "armband", "armbänder", "pulsera", "braccialetto"],
        "classify_patterns": [
            r"\bbracelets?\b",
            r"\barmband\w*",
            r"\bpulseras?\b",
            r"\bbraccialett[oi]\b",
            r"\barmbandje\b",
        ],
    },
    {
        "name": "ring",
        "parent": "jewelry",
        "search_tokens": ["ring", "rings", "ringe"],
        "classify_patterns": [
            r"\brings?\b",
            r"\bringe\b",
        ],
    },
    {
        "name": "watch",
        "parent": "jewelry",
        "search_tokens": ["watch", "watches", "uhr", "uhren", "montre", "reloj", "orologio", "horloge"],
        "classify_patterns": [
            # NOTE: smartwatch / fitness-watch already matched above and
            # claimed those products. This pattern catches mechanical /
            # quartz / style watches that sail past those.
            r"\bwatch(?:es)?\b",        # watch or watches (NOT watche)
            r"\buhr(?:en)?\b",          # uhr or uhren
            r"\bmontres?\b",
            r"\breloj(?:es)?\b",
            r"\borologi[oi]\b",         # orologio or orologii
            r"\bhorloges?\b",
        ],
    },

    # =================================================================
    # ACCESSORIES (style — hats, scarves, belts, gloves)
    # =================================================================
    {
        "name": "hat",
        "parent": "accessories",
        "search_tokens": ["hat", "hats", "cap", "caps", "beanie", "mütze", "chapeau", "sombrero"],
        "classify_patterns": [
            r"\bhats?\b", r"\bcaps?\b", r"\bbeanies?\b",
            r"\bm(?:ü|u|ue)tzen?\b",
            r"\bchapeaux?\b",
            r"\bsombreros?\b",
        ],
    },
    {
        "name": "scarf",
        "parent": "accessories",
        "search_tokens": ["scarf", "scarves", "schal", "écharpe", "bufanda"],
        "classify_patterns": [
            r"\bscarf\b", r"\bscarves\b",
            r"\bschals?\b",
            r"\b[ée]charpes?\b",
            r"\bbufandas?\b",
        ],
    },
    {
        "name": "belt",
        "parent": "accessories",
        "search_tokens": ["belt", "belts", "gürtel", "ceinture", "cinturón"],
        "classify_patterns": [
            r"\bbelts?\b",
            r"\bg(?:ü|u|ue)rtel\b",
            r"\bceintures?\b",
            r"\bcintur[oó]nes?\b", r"\bcintur[oó]n\b",
        ],
    },
    {
        "name": "glove",
        "parent": "accessories",
        "search_tokens": ["glove", "gloves", "handschuh", "gants"],
        "classify_patterns": [
            r"\bgloves?\b",
            r"\bhandschuh\w*",
            r"\bgants?\b",
        ],
    },

    # =================================================================
    # PHONE CASE — must precede BAGS.wallet so "MagSafe Wallet Phone
    # Case Leather" classifies as phone-case (the lead noun) instead of
    # wallet via the shared 'wallet' alternation.
    # =================================================================
    {
        "name": "phone-case",
        "parent": "electronics",
        "search_tokens": ["phone-case", "phone case", "iphone-case", "iphone case", "magsafe-case", "wallet-case"],
        "classify_patterns": [
            # Allow up to 4 words between the phone qualifier and 'case'
            # so "iPhone 15 Pro Magnetic Case" / "MagSafe Wallet Phone
            # Case Leather" / "Samsung Galaxy S24 Ultra Case" all hit.
            r"\b(?:phone|iphone|samsung|galaxy|magsafe)\b[\s\-]+(?:\w+[\s\-]+){0,4}cases?\b",
            r"\bmagsafe[\s\-]?(?:case|wallet)\w*",
            r"\bwallet[\s\-]?cases?\b",
            r"\bhandyh(?:ü|u|ue)lle\w*",
        ],
    },

    # =================================================================
    # BAGS
    # =================================================================
    {
        "name": "handbag",
        "parent": "bags",
        "search_tokens": ["handbag", "handbags", "purse", "purses"],
        "classify_patterns": [
            r"\bhandbags?\b",
            r"\bpurses?\b",
        ],
    },
    {
        "name": "backpack",
        "parent": "bags",
        "search_tokens": ["backpack", "backpacks", "rucksack", "sac à dos"],
        "classify_patterns": [
            r"\bbackpacks?\b",
            r"\brucks(?:a|ä)cke?\b",
            r"\bsac\s+(?:à|a)\s+dos\b",
        ],
    },
    {
        "name": "tote",
        "parent": "bags",
        "search_tokens": ["tote", "totes", "tote-bag"],
        "classify_patterns": [
            r"\btotes?\b",
            r"\btote[\s\-]?bags?\b",
        ],
    },
    {
        "name": "crossbody",
        "parent": "bags",
        "search_tokens": ["crossbody", "cross-body"],
        "classify_patterns": [
            r"\bcrossbody\b",
        ],
    },
    {
        "name": "clutch",
        "parent": "bags",
        "search_tokens": ["clutch", "clutches"],
        "classify_patterns": [
            r"\bclutches?\b",
        ],
    },
    {
        "name": "wallet",
        "parent": "bags",
        "search_tokens": ["wallet", "wallets", "portefeuille", "geldbörse", "cartera"],
        "classify_patterns": [
            r"\bwallets?\b",
            r"\bportefeuilles?\b",
            r"\bgeldb(?:ö|o|oe)rsen?\b",
            r"\bcarteras?\b",
        ],
    },
    {
        "name": "makeup-bag",
        "parent": "bags",
        "search_tokens": ["makeup-bag", "make-up bag", "cosmetic bag", "kulturbeutel"],
        "classify_patterns": [
            r"\bmake[\s\-]?up[\s\-]?bags?\b",
            r"\bcosmetic[\s\-]?bags?\b",
            r"\bkulturbeutel\b",
        ],
    },

    # =================================================================
    # COSTUMES
    # =================================================================
    {
        "name": "costume",
        "parent": "costumes",
        "search_tokens": ["costume", "costumes", "cosplay", "halloween-costume", "halloween costume"],
        "classify_patterns": [
            r"\bcostumes?\b", r"\bcosplay\w*",
            r"\bhalloween[\s\-]?(?:costume|outfit|wig)\w*",
            r"\bfancy[\s\-]?dress\b",
            r"\bmascot[\s\-]?costume\w*",
        ],
    },
    {
        "name": "halloween-mask",
        "parent": "costumes",
        "search_tokens": ["halloween-mask", "halloween mask", "latex-mask", "latex mask"],
        "classify_patterns": [
            r"\bhalloween[\s\-]?mask\w*",
            r"\blatex[\s\-]?mask\w*",
        ],
    },

    # =================================================================
    # ELECTRONICS — phone-case lives in its own block above (before
    # BAGS) so it precedes the wallet pattern.
    # =================================================================
    {
        "name": "headphones",
        "parent": "electronics",
        "search_tokens": ["headphones", "earbuds", "earphones", "kopfhörer"],
        "classify_patterns": [
            r"\bheadphones?\b",
            r"\bearbuds?\b",
            r"\bearphones?\b",
            r"\bkopfh(?:ö|o|oe)rer\b",
        ],
    },
    {
        "name": "charger",
        "parent": "electronics",
        "search_tokens": ["charger", "power-bank", "power bank", "powerbank", "ladegerät"],
        "classify_patterns": [
            r"\bchargers?\b",
            r"\bpower[\s\-]?banks?\b",
            r"\bladeger(?:ä|a|ae)t\w*",
        ],
    },
    {
        "name": "smart-camera",
        "parent": "electronics",
        "search_tokens": ["security-camera", "wifi camera", "wireless camera"],
        "classify_patterns": [
            r"\b(?:security|wifi|wireless|hidden|spy|surveillance)[\s\-]?cameras?\b",
            r"\bbackup[\s\-]?cameras?\b",
        ],
    },
    {
        "name": "ssd",
        "parent": "electronics",
        "search_tokens": ["ssd", "external ssd", "hard drive", "external hard drive"],
        "classify_patterns": [
            r"\bssd\b", r"\bexternal[\s\-]?ssd\b",
            r"\bhard[\s\-]?drives?\b",
            r"\bexternal[\s\-]?storage\b",
        ],
    },

    # =================================================================
    # OUTDOORS / UTILITY (handhelds, walking sticks, etc.)
    # =================================================================
    {
        "name": "trekking-pole",
        "parent": "outdoors",
        "search_tokens": ["trekking-pole", "trekking pole", "hiking-pole", "walking-stick", "wanderstock"],
        "classify_patterns": [
            r"\btrekking[\s\-]?poles?\b",
            r"\bhiking[\s\-]?poles?\b",
            r"\bwalking[\s\-]?sticks?\b",
            r"\bwanderstock\w*",
            r"\bspazierstock\w*",
            r"\bself[\s\-]?defen[sc]e[\s\-]?(?:walking[\s\-]?)?sticks?\b",
        ],
    },
    {
        "name": "magnifying-glass",
        "parent": "outdoors",
        "search_tokens": ["magnifying-glass", "magnifying glass", "magnifier", "lupe"],
        "classify_patterns": [
            r"\bmagnifying[\s\-]?glass\w*",
            r"\bhand[\s\-]?free[\s\-]?magnif\w*",
            r"\blupe\b", r"\blupen\b",
            r"\blesehilfe\w*",
        ],
    },
    {
        "name": "umbrella",
        "parent": "outdoors",
        "search_tokens": ["umbrella", "umbrellas", "regenschirm", "parapluie", "paraguas", "ombrello"],
        "classify_patterns": [
            r"\bumbrellas?\b",
            r"\bregenschirm\w*",
            r"\bparapluies?\b",
            r"\bparaguas\b",
            r"\bombrellos?\b",
            r"\bparaplus?\b",
        ],
    },

    # =================================================================
    # BEAUTY (function devices)
    # =================================================================
    {
        "name": "snap-on-veneers",
        "parent": "beauty",
        "search_tokens": ["snap-on-veneers", "cosmetic-veneers", "snap on veneers"],
        "classify_patterns": [
            r"\bsnap[\s\-]?on[\s\-]?veneers?\b",
            r"\bcosmetic[\s\-]?veneers?\b",
        ],
    },
    {
        "name": "face-massager",
        "parent": "beauty",
        "search_tokens": ["face-massager", "facial-massager", "beauty-wand"],
        "classify_patterns": [
            r"\bfacial?[\s\-]?massagers?\b",
            r"\bbeauty[\s\-]?wands?\b",
        ],
    },
    {
        "name": "hair-tool",
        "parent": "beauty",
        "search_tokens": ["hair-clipper", "hair clipper", "curling-iron", "hair brush", "hair-dryer"],
        "classify_patterns": [
            r"\bhair[\s\-]?(?:clipper|trimmer|brush|dryer|straighten)\w*",
            r"\bcurling[\s\-]?iron\w*",
            r"\bhair[\s\-]?flex\w*",
        ],
    },

    # =================================================================
    # HOME (decor, kitchen, bedding)
    # =================================================================
    {
        "name": "blanket",
        "parent": "home",
        "search_tokens": ["blanket", "blankets", "decke", "throw"],
        "classify_patterns": [
            r"\bblankets?\b", r"\bthrow\b",
            r"\bdecken?\b",
            r"\bweighted[\s\-]?blanket\w*",
        ],
    },
    {
        "name": "pillow",
        "parent": "home",
        "search_tokens": ["pillow", "pillows", "kissen", "cushion"],
        "classify_patterns": [
            r"\bpillows?\b",
            r"\bcushions?\b",
            r"\bkissen?\b",
        ],
    },
    {
        "name": "vase",
        "parent": "home",
        "search_tokens": ["vase", "vases", "vaas"],
        "classify_patterns": [
            r"\bvases?\b",
        ],
    },
    {
        "name": "candle",
        "parent": "home",
        "search_tokens": ["candle", "candles", "kerze"],
        "classify_patterns": [
            r"\bcandles?\b",
            r"\bkerzen?\b",
        ],
    },
]


# ===================================================================
# Compile classify regex per category for fast lookup at scrape time.
# ===================================================================
_COMPILED: list[tuple[str, str, re.Pattern]] = []
for _cat in PRODUCT_CATEGORIES:
    _name = _cat["name"]
    _parent = _cat["parent"]
    _alts = "|".join(f"(?:{p})" for p in _cat["classify_patterns"])
    _COMPILED.append((_name, _parent, re.compile(_alts, re.IGNORECASE)))


def assign_product_category(
    title: str = "",
    handle: str = "",
    image_url: str = "",
    product_type: str = "",
) -> str:
    """Assign the most-specific category from PRODUCT_CATEGORIES that
    matches any of the input fields. Returns "" if nothing matches.

    Iteration is top-down through the catalog, so wearable-gadget
    categories (smartwatch, posture-corrector, dog-collar) win over
    the apparel/jewelry blocks that come later. This mirrors the
    FORCE_GENERAL precedence in scraper.py.
    """
    blob = " ".join([title or "", handle or "", image_url or "", product_type or ""])
    for name, _parent, pattern in _COMPILED:
        if pattern.search(blob):
            return name
    return ""


# ===================================================================
# Search index: token → set of category names.
# ===================================================================
# Build TOKEN_TO_CATEGORIES from BOTH every search_token in the
# catalog AND every parent name (so "lighting" returns all lighting
# children). Singular and naive plural forms are added automatically
# so "chandeliers" works as well as "chandelier".
PARENTS_TO_CHILDREN: dict[str, set[str]] = {}
TOKEN_TO_CATEGORIES: dict[str, set[str]] = {}
ALL_CATEGORY_NAMES: set[str] = set()


def _norm(token: str) -> str:
    return token.strip().lower()


def _plural_variants(token: str) -> set[str]:
    """Naive English plural / singular variants. Catches the common
    cases (chandelier ↔ chandeliers, dress ↔ dresses, hoop ↔ hoops)
    without dragging in an NLP library."""
    t = _norm(token)
    out = {t}
    if len(t) > 3:
        if t.endswith("ies"):
            out.add(t[:-3] + "y")          # categories → category
        elif t.endswith("es") and not t.endswith("oes"):
            out.add(t[:-2])                 # dresses → dress
            out.add(t[:-1])                 # dresses → dresse (plural noise — harmless)
        elif t.endswith("s") and not t.endswith("ss"):
            out.add(t[:-1])                 # chandeliers → chandelier
        else:
            out.add(t + "s")
            out.add(t + "es")
    return out


def _index_token(token: str, category: str) -> None:
    for v in _plural_variants(token):
        TOKEN_TO_CATEGORIES.setdefault(v, set()).add(category)


for _cat in PRODUCT_CATEGORIES:
    _name = _cat["name"]
    _parent = _cat["parent"]
    ALL_CATEGORY_NAMES.add(_name)
    PARENTS_TO_CHILDREN.setdefault(_parent, set()).add(_name)
    # The category's own slug is searchable.
    _index_token(_name, _name)
    # Each multilingual search token resolves to this category.
    for tok in _cat["search_tokens"]:
        _index_token(tok, _name)
        # Hyphenated tokens also indexed in their space-separated form.
        if "-" in tok:
            _index_token(tok.replace("-", " "), _name)
            _index_token(tok.replace("-", ""), _name)
        if " " in tok:
            _index_token(tok.replace(" ", "-"), _name)
            _index_token(tok.replace(" ", ""), _name)

# Parents are searchable too — searching "lighting" returns every
# child category under that umbrella.
for parent, children in PARENTS_TO_CHILDREN.items():
    if not parent:
        continue
    _index_token(parent, parent)
    # Tag the parent name as a "virtual" category that expands to all
    # children at search time — keep it distinguishable by storing the
    # parent's set of child categories under a special _parent_ prefix.
    TOKEN_TO_CATEGORIES.setdefault(parent, set())
    TOKEN_TO_CATEGORIES[parent].update(children)


# ===================================================================
# Public lookup helpers used by main.py search.
# ===================================================================
def lookup_categories_for_query_token(token: str) -> set[str]:
    """Return the set of canonical product categories that the user's
    search token implies. Handles plural/singular and the parent
    expansion (typing 'lighting' returns all lighting children).

    Empty set if the token isn't a recognised category-noun. The
    search then falls back to title/ai_tags substring matching.
    """
    t = _norm(token)
    if not t:
        return set()
    if t in TOKEN_TO_CATEGORIES:
        return set(TOKEN_TO_CATEGORIES[t])
    for v in _plural_variants(t):
        if v in TOKEN_TO_CATEGORIES:
            return set(TOKEN_TO_CATEGORIES[v])
    return set()


def is_category_token(token: str) -> bool:
    return bool(lookup_categories_for_query_token(token))
