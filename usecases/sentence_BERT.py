from sentence_transformers import SentenceTransformer
import numpy as np

# 加载 Universal Sentence Encoder 模型
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 输入句子
sentences = [
    "place of birth",
    "has part(s)",
    "award received",
    "languages spoken, written or signed",
    "cast member",
    "genre",
    "educated at",
    "distribution format",
    "image",
    "IMDb ID",
    "filming location",
    "director",
    "Kijkwijzer rating",
    "MPA film rating",
    "RCQ classification",
    "country",
    "performer",
    "screenwriter",
    "based on",
    "father",
    "original language of film or TV show",
    "described by source",
    "notable work",
    "instance of",
    "sport",
    "country of citizenship",
    "occupation",
    "CNC film rating (France)",
    "spouse",
    "distributed by",
    "language of work or name",
    "diplomatic relation",
    "place of death",
    "country of origin",
    "contributor to the creative work or subject",
    "JMK film rating",
    "located in/on physical feature",
    "CNC film rating (Romania)",
    "FSK film rating",
    "narrative location",
    "main subject",
    "participant in",
    "manner of death",
    "twinned administrative body",
    "animator",
    "publication date",
    "costume designer",
    "residence",
    "ClassInd rating",
    "voice actor",
    "has works in the collection",
    "director of photography",
    "present in work",
    "production company",
    "film editor",
    "industry",
    "media franchise",
    "make-up artist",
    "time period",
    "owner of",
    "nominated for",
    "color",
    "different from",
    "Filmiroda rating",
    "ICAA rating",
    "production designer",
    "characters",
    "OFLC classification",
    "derivative work",
    "owned by",
    "continent",
    "native language",
    "work location",
    "part of",
    "child",
    "executive producer",
    "unmarried partner",
    "country for sport",
    "killed by",
    "publisher",
    "headquarters location",
    "official language",
    "place served by transport hub",
    "has characteristic",
    "assessment",
    "Medierådet rating",
    "MTRCB rating",
    "creator",
    "winner",
    "place of burial",
    "film crew member",
    "relative",
    "dedicated to",
    "shares border with",
    "ethnic group",
    "employer",
    "religion or worldview",
    "scenographer",
    "set in environment",
    "platform",
    "significant event",
    "set in period",
    "writing language",
    "BAMID film rating",
    "BBFC rating",
    "subclass of",
    "influenced by",
    "sibling",
    "from narrative universe",
    "location",
    "after a work by",
    "conflict",
    "season",
    "RTC film rating",
    "participant",
    "FPB rating",
    "capital of",
    "followed by",
    "NMHH film rating",
    "author",
    "form of creative work",
    "follows",
    "convicted of",
    "founded by",
    "developer",
    "given name",
    "depicts",
    "narrator",
    "takes place in fictional universe",
    "aspect ratio (W:H)",
    "first appearance",
    "part of the series",
    "inspired by",
    "storyboard artist",
    "collection",
    "military branch",
    "art director",
    "EIRIN film rating",
    "RARS rating",
    "fabrication method",
    "historic county",
    "cause of death",
    "member of",
    "uses",
    "replaced by",
    "set during recurring event",
    "movement",
    "fictional universe described in",
    "INCAA film rating",
    "said to be the same as",
    "partner in business or sport",
    "location of formation",
    "field of this occupation",
    "box office",
    "KAVI rating",
    "significant person",
    "head of state",
    "original broadcaster",
    "capital",
    "facet of",
    "located in the administrative territorial entity",
    "sound designer",
    "parent organization",
    "original film format",
    "product or material produced or sold, or service provided",
    "located in or next to body of water",
    "sexual orientation",
    "IGAC rating",
    "sex or gender",
    "choreographer",
    "operating system",
    "enemy",
    "presented in",
    "copyright status",
    "contains the administrative territorial entity",
    "language used",
    "named after",
    "place of publication",
    "home world",
    "references work, tradition or theory",
    "IFCO rating",
    "operating area",
    "crew member(s)",
    "character designer",
    "field of work",
    "opposite of",
    "student",
    "mother",
    "IMDA rating",
    "health specialty",
    "allegiance",
    "broadcast by",
    "student of",
    "business model",
    "indigenous to",
    "list of works",
    "sidekick of",
    "medical condition",
    "located in the present-day administrative territorial entity",
    "director / manager",
    "archives at",
    "partially coincident with",
    "operator",
    "basin country",
    "stepparent",
    "conferred by",
    "superhuman feature or ability",
    "interested in",
    "intended public",
    "narrative motif",
    "ancestral home",
    "Hong Kong film rating",
    "public holiday",
    "sports discipline competed in",
    "copyright holder",
    "affiliation",
    "has edition or translation",
    "depicted by",
    "member of the crew of",
    "musical conductor",
    "replaces",
    "lifestyle",
    "permanent resident of",
    "presenter",
    "head of government",
    "official residence",
    "KMRB film rating",
    "represented by",
    "has pet",
    "applies to jurisdiction",
    "designed by",
    "has cause",
    "has use",
    "Australian Classification",
    "edition or translation of",
    "cites work",
    "occupant",
    "input device",
    "copyright representative",
    "practiced by",
    "noble title",
    "plot expanded in",
    "quotes work",
    "political ideology",
    "has effect",
    "lowest point",
    "member of sports team",
    "place of detention",
    "located on street",
    "copyright license",
]

sentence_embeddings = model.encode(sentences)
norms_b = np.linalg.norm(sentence_embeddings, axis=1)

def get_closest(relation):
    relation_embeddings = model.encode(relation)
    norm_a = np.linalg.norm(relation_embeddings)
    cosine_similarities = np.dot(sentence_embeddings, relation_embeddings) / (norm_a * norms_b)
    return sentences[np.argmax(cosine_similarities)]

if __name__ == '__main__':
    get_closest("MPAA_rating")