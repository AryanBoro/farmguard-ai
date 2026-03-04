"""
FarmGuard AI — Full Remedy & Treatment Database
Covers all 38 plant disease classes from PlantVillage dataset
"""

REMEDY_DB = {
    # ─── APPLE ────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "common_name": "Apple Scab",
        "pathogen": "Venturia inaequalis (fungus)",
        "severity": "moderate",
        "description": "Olive-green to brown velvety spots on leaves and fruit. Leaves may curl and drop early.",
        "immediate_actions": [
            "Remove and destroy all infected leaves and fallen debris immediately.",
            "Apply a fungicide (captan, myclobutanil, or mancozeb) at 7–10 day intervals during wet periods.",
            "Prune to improve canopy airflow."
        ],
        "prevention": [
            "Plant scab-resistant apple varieties (e.g. Liberty, Enterprise).",
            "Apply dormant copper sprays before bud break each spring.",
            "Maintain clean orchard floor — rake and compost fallen leaves in autumn."
        ],
        "organic_options": ["Neem oil sprays", "Sulfur-based fungicides (avoid in high heat)", "Lime sulfur dormant sprays"],
        "risk_factors": ["Prolonged leaf wetness (>9 hrs)", "Temperatures 13–24°C during budbreak", "Dense planting"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "13-24°C"}
    },
    "Apple___Black_rot": {
        "common_name": "Apple Black Rot",
        "pathogen": "Botryosphaeria obtusa (fungus)",
        "severity": "high",
        "description": "Circular brown lesions on fruit expanding into black rot. Frogeye leaf spots (purple margin, tan center) on leaves.",
        "immediate_actions": [
            "Prune and destroy all cankered wood, mummified fruit, and infected branches.",
            "Apply captan or thiophanate-methyl fungicide starting at petal fall.",
            "Remove dead wood that harbors the pathogen year-round."
        ],
        "prevention": [
            "Keep trees vigorous through balanced fertilization — stressed trees are more susceptible.",
            "Eliminate wild or abandoned apple trees nearby (reservoir hosts).",
            "Apply protective fungicide sprays every 10–14 days from pink stage through harvest."
        ],
        "organic_options": ["Copper-based sprays", "Neem oil", "Bordeaux mixture"],
        "risk_factors": ["Wounded or stressed trees", "Warm wet weather (24–29°C)", "Poor pruning leaving stubs"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "24-29°C"}
    },
    "Apple___Cedar_apple_rust": {
        "common_name": "Cedar Apple Rust",
        "pathogen": "Gymnosporangium juniperi-virginianae (fungus)",
        "severity": "moderate",
        "description": "Bright orange-yellow spots on upper leaf surfaces; tube-like structures underneath. Requires cedar/juniper as alternate host.",
        "immediate_actions": [
            "Apply myclobutanil or propiconazole fungicide at pink bud stage, repeating every 7–10 days for 3 applications.",
            "Remove nearby Eastern red cedar or juniper trees if feasible."
        ],
        "prevention": [
            "Plant rust-resistant apple varieties.",
            "Establish buffer distance from juniper plantings.",
            "Begin fungicide program early — infection happens during bloom."
        ],
        "organic_options": ["Sulfur fungicides (applied preventively)", "Neem oil has limited efficacy"],
        "risk_factors": ["Proximity to Eastern red cedar", "Wet spring weather during bloom", "Temperatures 8–24°C"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "8-24°C"}
    },
    "Apple___healthy": {
        "common_name": "Healthy Apple",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy with no visible disease symptoms.",
        "immediate_actions": ["No treatment required. Continue current management."],
        "prevention": [
            "Maintain regular scouting schedule (weekly during growing season).",
            "Keep records of spray timings and weather conditions.",
            "Ensure balanced nutrition — avoid excess nitrogen which promotes soft growth."
        ],
        "organic_options": [],
        "risk_factors": ["Monitor during prolonged wet periods", "Watch for early scab/rust at bud break"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── BLUEBERRY ────────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "common_name": "Healthy Blueberry",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy with no visible disease symptoms.",
        "immediate_actions": ["No treatment required."],
        "prevention": [
            "Maintain soil pH between 4.5–5.5 for optimal health.",
            "Mulch with pine bark to conserve moisture and suppress weeds.",
            "Scout regularly for mummy berry and botrytis during bloom."
        ],
        "organic_options": [],
        "risk_factors": ["Alkaline soil stress", "Overhead irrigation promoting fungal issues"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── CHERRY ───────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "common_name": "Cherry Powdery Mildew",
        "pathogen": "Podosphaera clandestina (fungus)",
        "severity": "moderate",
        "description": "White powdery coating on young leaves, shoots, and fruit. Infected leaves may curl and turn brown.",
        "immediate_actions": [
            "Apply sulfur or potassium bicarbonate fungicide at first sign of infection.",
            "Remove and destroy heavily infected shoots.",
            "Improve air circulation through selective pruning."
        ],
        "prevention": [
            "Avoid excess nitrogen fertilization which encourages soft succulent growth.",
            "Apply preventive sulfur sprays from bud swell through shoot elongation.",
            "Plant in full sun with good air drainage."
        ],
        "organic_options": ["Potassium bicarbonate", "Neem oil", "Sulfur dust or spray"],
        "risk_factors": ["High humidity without rain", "Shaded dense canopy", "High nitrogen levels", "Temperatures 20–27°C"],
        "weather_risk": {"high_humidity": True, "rain_triggered": False, "temp_range": "20-27°C"}
    },
    "Cherry_(including_sour)___healthy": {
        "common_name": "Healthy Cherry",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Regular pruning for airflow", "Preventive copper sprays in autumn"],
        "organic_options": [],
        "risk_factors": ["Monitor for brown rot during fruit ripening"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── CORN ─────────────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "common_name": "Corn Gray Leaf Spot",
        "pathogen": "Cercospora zeae-maydis (fungus)",
        "severity": "high",
        "description": "Rectangular, tan to gray lesions with distinct parallel edges running between leaf veins. Severely reduces photosynthesis.",
        "immediate_actions": [
            "Apply triazole or strobilurin fungicide (e.g. pyraclostrobin, propiconazole) at VT/R1 stage.",
            "Improve field drainage and reduce tillage debris."
        ],
        "prevention": [
            "Plant resistant or tolerant hybrids — single most effective strategy.",
            "Rotate with non-host crops (soybean, wheat) for at least one season.",
            "Bury crop residue through tillage to reduce overwintering inoculum."
        ],
        "organic_options": ["Copper-based sprays (limited efficacy)", "Crop rotation is primary organic strategy"],
        "risk_factors": ["Minimum tillage retaining residue", "High humidity and dew", "Dense planting reducing airflow"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "25-30°C"}
    },
    "Corn_(maize)___Common_rust_": {
        "common_name": "Corn Common Rust",
        "pathogen": "Puccinia sorghi (fungus)",
        "severity": "moderate",
        "description": "Small, oval, brick-red pustules scattered on both leaf surfaces. Pustules rupture releasing powdery rust-colored spores.",
        "immediate_actions": [
            "Apply strobilurin fungicide (azoxystrobin) or triazole at early rust detection.",
            "Fungicide most effective applied before tasseling."
        ],
        "prevention": [
            "Use rust-resistant hybrid varieties.",
            "Scout fields weekly from V6 stage onward.",
            "Avoid late planting that exposes crop to high-spore-load periods."
        ],
        "organic_options": ["Sulfur sprays", "Neem oil (preventive only)"],
        "risk_factors": ["Cool temperatures 16–23°C with high humidity", "Airborne spore dispersal — wind spread", "Susceptible hybrids"],
        "weather_risk": {"high_humidity": True, "rain_triggered": False, "temp_range": "16-23°C"}
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "common_name": "Northern Corn Leaf Blight",
        "pathogen": "Exserohilum turcicum (fungus)",
        "severity": "high",
        "description": "Long (5–15 cm) cigar-shaped, grayish-green to tan lesions on leaves. Can cause significant yield loss if infection occurs before tasseling.",
        "immediate_actions": [
            "Apply triazole or strobilurin fungicide at first sign, particularly before tasseling.",
            "Prioritize fields with susceptible hybrids and residue from previous corn crop."
        ],
        "prevention": [
            "Select resistant or moderately resistant hybrids.",
            "Rotate crops and incorporate residue to reduce inoculum.",
            "Avoid overhead irrigation late in the day."
        ],
        "organic_options": ["Copper fungicides", "Bacillus subtilis biocontrol sprays"],
        "risk_factors": ["Moderate temperatures 18–27°C", "Extended leaf wetness periods", "Corn-on-corn rotation"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "18-27°C"}
    },
    "Corn_(maize)___healthy": {
        "common_name": "Healthy Corn",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Scout weekly from V6", "Ensure balanced potassium for disease resistance"],
        "organic_options": [],
        "risk_factors": ["Monitor during humid periods", "Check for rust pustules on lower leaves first"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── GRAPE ────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "common_name": "Grape Black Rot",
        "pathogen": "Guignardia bidwellii (fungus)",
        "severity": "high",
        "description": "Circular tan leaf lesions with dark borders; infected berries turn brown then shrivel into hard black mummies.",
        "immediate_actions": [
            "Remove and destroy all mummified fruit immediately — primary inoculum source.",
            "Apply myclobutanil or mancozeb fungicide starting at bud break, every 10–14 days.",
            "Ensure good spray coverage of all fruit clusters."
        ],
        "prevention": [
            "Remove mummies from vine and ground before dormant season.",
            "Prune for open canopy with maximum air and light penetration.",
            "Begin fungicide program at 1-inch shoot growth and continue through veraison."
        ],
        "organic_options": ["Bordeaux mixture", "Copper hydroxide sprays"],
        "risk_factors": ["Warm (26–32°C) wet weather during bloom", "Retained mummified fruit", "Dense canopy"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "26-32°C"}
    },
    "Grape___Esca_(Black_Measles)": {
        "common_name": "Grape Esca (Black Measles)",
        "pathogen": "Phaeomoniella chlamydospora & Phaeoacremonium spp. (fungal complex)",
        "severity": "high",
        "description": "Tiger-stripe pattern on leaves (interveinal necrosis); berries develop dark spots. Internal wood shows brown streaking. Chronic and acute (apoplexy) forms exist.",
        "immediate_actions": [
            "No curative chemical treatment exists. Remove and destroy severely affected vines.",
            "Paint pruning wounds immediately with wound sealant or fungicide paste.",
            "Avoid large pruning cuts during wet weather."
        ],
        "prevention": [
            "Use double-pruning technique — leave a spur, prune to final position weeks later when wood is dry.",
            "Sterilize pruning tools with 70% alcohol or 10% bleach between vines.",
            "Avoid water stress which increases susceptibility."
        ],
        "organic_options": ["Trichoderma-based wound protectants", "Boric acid paste on wounds"],
        "risk_factors": ["Large pruning wounds in wet weather", "Old vines (>10 years)", "Drought stress"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "any"}
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "common_name": "Grape Isariopsis Leaf Spot",
        "pathogen": "Pseudocercospora vitis (fungus)",
        "severity": "low",
        "description": "Angular reddish-brown spots on upper leaf surface; grayish sporulation on undersides. Rarely causes severe economic damage.",
        "immediate_actions": [
            "Apply copper-based or mancozeb fungicide if infection is severe.",
            "Remove heavily infected leaves."
        ],
        "prevention": [
            "Maintain open canopy for airflow.",
            "Avoid overhead irrigation.",
            "Standard fungicide program for downy mildew often provides adequate protection."
        ],
        "organic_options": ["Copper sprays", "Neem oil"],
        "risk_factors": ["High humidity", "Warm temperatures", "Overhead irrigation"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "20-30°C"}
    },
    "Grape___healthy": {
        "common_name": "Healthy Grape",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Regular canopy management", "Begin protective spray program at bud swell"],
        "organic_options": [],
        "risk_factors": ["Monitor bloom period closely for downy/powdery mildew"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── ORANGE ───────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "common_name": "Citrus Greening (HLB)",
        "pathogen": "Candidatus Liberibacter asiaticus (bacteria) — spread by Asian citrus psyllid",
        "severity": "critical",
        "description": "Yellow shoots (huanglongbing = 'yellow dragon disease'). Asymmetric blotchy yellowing of leaves, small lopsided bitter fruit, stunted growth. Currently incurable — kills trees within years.",
        "immediate_actions": [
            "IMMEDIATELY notify local agricultural extension or plant health authority.",
            "Do NOT move plant material off property — quarantine strictly.",
            "Apply systemic insecticides (imidacloprid soil drench or foliar) to kill psyllid vector immediately.",
            "Remove and destroy confirmed infected trees to protect neighbors."
        ],
        "prevention": [
            "Use certified disease-free nursery stock only.",
            "Establish regular psyllid monitoring and control program.",
            "Apply reflective mulches to deter psyllids.",
            "Nutritional programs (foliar zinc, micronutrients) can extend productive life of mildly infected trees."
        ],
        "organic_options": ["Kaolin clay barrier sprays to deter psyllids", "Beneficial insect programs for psyllid biocontrol"],
        "risk_factors": ["Presence of Asian citrus psyllid", "Proximity to infected orchards", "Warm climates year-round"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── PEACH ────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "common_name": "Peach Bacterial Spot",
        "pathogen": "Xanthomonas arboricola pv. pruni (bacteria)",
        "severity": "moderate",
        "description": "Small, water-soaked spots on leaves turning brown with yellow halos; spots may fall out leaving 'shot holes'. Fruit shows pitting and cracking.",
        "immediate_actions": [
            "Apply copper bactericide (copper hydroxide or copper octanoate) during early season.",
            "Avoid overhead irrigation — use drip irrigation.",
            "Remove severely infected leaves and shoots."
        ],
        "prevention": [
            "Plant resistant varieties (Contender, Redhaven have moderate resistance).",
            "Apply copper sprays at shuck split and every 10–14 days in wet weather.",
            "Avoid excessive nitrogen fertilization which produces susceptible soft tissue."
        ],
        "organic_options": ["Copper-based bactericides", "Bacillus subtilis (limited efficacy)"],
        "risk_factors": ["Windy rainy conditions that create wounds", "Warm temperatures 24–30°C", "Susceptible varieties"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "24-30°C"}
    },
    "Peach___healthy": {
        "common_name": "Healthy Peach",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Apply dormant copper spray before bud swell", "Thin fruit for air circulation"],
        "organic_options": [],
        "risk_factors": ["Monitor for bacterial spot during wet spring weather"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── PEPPER ───────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "common_name": "Pepper Bacterial Spot",
        "pathogen": "Xanthomonas euvesicatoria (bacteria)",
        "severity": "moderate",
        "description": "Small, water-soaked lesions on leaves that turn brown with yellow halos. Raised scabby spots on fruit. Defoliation reduces yield and exposes fruit to sunscald.",
        "immediate_actions": [
            "Apply copper + mancozeb tank mix immediately — copper alone can select for resistant strains.",
            "Remove and bag infected plant debris.",
            "Avoid working in field when plants are wet."
        ],
        "prevention": [
            "Use certified disease-free seed and transplants.",
            "Apply preventive copper sprays from transplanting through fruit set.",
            "Use drip irrigation and plastic mulch to reduce soil splash."
        ],
        "organic_options": ["Copper hydroxide", "Acibenzolar-S-methyl (plant activator)", "Bacillus subtilis"],
        "risk_factors": ["Splashing rain and overhead irrigation", "Warm (24–30°C) wet weather", "Infected seed"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "24-30°C"}
    },
    "Pepper,_bell___healthy": {
        "common_name": "Healthy Bell Pepper",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Use drip irrigation", "Scout for bacterial spot after rain events"],
        "organic_options": [],
        "risk_factors": ["Bacterial spot risk increases after storms"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── POTATO ───────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "common_name": "Potato Early Blight",
        "pathogen": "Alternaria solani (fungus)",
        "severity": "moderate",
        "description": "Dark brown lesions with concentric rings (target board pattern) on older/lower leaves first. Yellow halo may surround lesions. Premature defoliation reduces tuber yield.",
        "immediate_actions": [
            "Apply chlorothalonil, mancozeb, or azoxystrobin fungicide at first sign.",
            "Increase spray frequency to every 7 days during warm wet weather.",
            "Remove heavily infected lower leaves."
        ],
        "prevention": [
            "Use certified disease-free seed potatoes.",
            "Rotate crops — do not plant potato/tomato in same spot for 2–3 years.",
            "Maintain adequate fertility — potassium especially helps disease resistance.",
            "Begin preventive fungicide program when plants reach 30 cm height."
        ],
        "organic_options": ["Copper-based fungicides", "Bacillus subtilis (Serenade)", "Neem oil"],
        "risk_factors": ["Plant stress (drought, poor nutrition)", "Warm days (24–29°C) with cool nights causing dew", "Dense canopy"],
        "weather_risk": {"high_humidity": True, "rain_triggered": False, "temp_range": "24-29°C"}
    },
    "Potato___Late_blight": {
        "common_name": "Potato Late Blight",
        "pathogen": "Phytophthora infestans (oomycete)",
        "severity": "critical",
        "description": "Water-soaked, pale green lesions that rapidly turn brown-black with white sporulation on leaf undersides in humid conditions. Can destroy entire field within days. Caused the Irish Famine of 1845.",
        "immediate_actions": [
            "Apply systemic fungicide (metalaxyl/mefenoxam or cymoxanil) + contact fungicide IMMEDIATELY.",
            "Spray in the evening to maximize contact time before morning dew burns off.",
            "Destroy (burn or bury) any heavily infected plant material — do NOT compost.",
            "Alert neighboring farms — airborne spores spread rapidly."
        ],
        "prevention": [
            "Plant resistant varieties (Sarpo Mira, Defender).",
            "Apply preventive contact fungicides (chlorothalonil, mancozeb) weekly during high-risk periods.",
            "Monitor weather — use late blight forecasting tools (BLITECAST, simPhyt).",
            "Destroy volunteer potato plants that harbor overwintering inoculum.",
            "Ensure good haulm destruction before harvest."
        ],
        "organic_options": ["Copper fungicides (bordeaux mixture, copper hydroxide) — most effective organic option", "Biofungicides have limited efficacy against late blight"],
        "risk_factors": ["Cool (10–21°C) wet weather with high humidity (>90%)", "Airborne spread from infected fields", "Susceptible varieties"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "10-21°C"}
    },
    "Potato___healthy": {
        "common_name": "Healthy Potato",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Monitor late blight forecasting services", "Apply preventive copper in high-risk weather"],
        "organic_options": [],
        "risk_factors": ["Late blight risk spikes dramatically during cool wet weather"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── RASPBERRY ────────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "common_name": "Healthy Raspberry",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Prune out old floricanes after harvest", "Thin new canes to 15–20 cm spacing for airflow"],
        "organic_options": [],
        "risk_factors": ["Monitor for botrytis during fruit ripening in wet weather"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── SOYBEAN ──────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "common_name": "Healthy Soybean",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Scout for sudden death syndrome and white mold during reproductive stages"],
        "organic_options": [],
        "risk_factors": ["Monitor for frogeye leaf spot and soybean rust in humid regions"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── SQUASH ───────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "common_name": "Squash Powdery Mildew",
        "pathogen": "Podosphaera xanthii / Erysiphe cichoracearum (fungi)",
        "severity": "moderate",
        "description": "White powdery spots on upper leaf surfaces, spreading to cover entire leaf. Infected leaves yellow and die prematurely.",
        "immediate_actions": [
            "Apply potassium bicarbonate, neem oil, or sulfur fungicide immediately.",
            "Remove and destroy heavily infected leaves.",
            "Increase plant spacing for better air circulation."
        ],
        "prevention": [
            "Plant resistant varieties where available.",
            "Apply preventive neem or sulfur sprays weekly in high-risk conditions.",
            "Avoid overhead irrigation — water in morning so foliage dries quickly."
        ],
        "organic_options": ["Potassium bicarbonate", "Neem oil", "Baking soda + horticultural oil spray", "Sulfur dust"],
        "risk_factors": ["High humidity without rain", "Warm dry days and cool nights", "Dense planting"],
        "weather_risk": {"high_humidity": True, "rain_triggered": False, "temp_range": "20-28°C"}
    },

    # ─── STRAWBERRY ───────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "common_name": "Strawberry Leaf Scorch",
        "pathogen": "Diplocarpon earlianum (fungus)",
        "severity": "moderate",
        "description": "Small, irregular purple spots on leaves that coalesce; leaves appear scorched and may die. Overwinters in infected plant debris.",
        "immediate_actions": [
            "Remove and destroy infected leaves.",
            "Apply captan or myclobutanil fungicide.",
            "Avoid overhead irrigation; use drip systems."
        ],
        "prevention": [
            "Use certified disease-free planting stock.",
            "Renovate plantings annually by mowing leaves after harvest and applying fungicide.",
            "Maintain wide row spacing for air circulation.",
            "Apply fungicide preventively in spring before symptoms appear."
        ],
        "organic_options": ["Copper-based fungicides", "Neem oil"],
        "risk_factors": ["Overhead irrigation", "Dense canopy", "Wet spring conditions"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "15-25°C"}
    },
    "Strawberry___healthy": {
        "common_name": "Healthy Strawberry",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy.",
        "immediate_actions": ["No treatment required."],
        "prevention": ["Renovate beds after harvest", "Use drip irrigation"],
        "organic_options": [],
        "risk_factors": ["Monitor for botrytis during fruit ripening"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },

    # ─── TOMATO ───────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "common_name": "Tomato Bacterial Spot",
        "pathogen": "Xanthomonas perforans / X. euvesicatoria (bacteria)",
        "severity": "moderate",
        "description": "Small, water-soaked spots on leaves turning brown with yellow halos; scabby raised spots on fruit. Defoliation reduces yield and causes fruit sunscald.",
        "immediate_actions": [
            "Apply copper + mancozeb mix immediately and every 5–7 days in wet weather.",
            "Remove infected plant material; disinfect tools.",
            "Avoid all field work when foliage is wet."
        ],
        "prevention": [
            "Use certified disease-free transplants and seed.",
            "Use drip irrigation exclusively.",
            "Apply acibenzolar-S-methyl (Actigard) as plant resistance inducer.",
            "Rotate with non-solanaceous crops for 2+ years."
        ],
        "organic_options": ["Copper hydroxide", "Bacillus subtilis (Serenade)", "Acibenzolar-S-methyl"],
        "risk_factors": ["Splashing rain and warm (24–30°C) temperatures", "Overhead irrigation", "Infected transplants"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "24-30°C"}
    },
    "Tomato___Early_blight": {
        "common_name": "Tomato Early Blight",
        "pathogen": "Alternaria solani (fungus)",
        "severity": "moderate",
        "description": "Dark concentric-ring lesions (target pattern) on older lower leaves first. Progresses up plant. Stem lesions ('collar rot') can girdle transplants.",
        "immediate_actions": [
            "Apply chlorothalonil, mancozeb, or azoxystrobin fungicide.",
            "Remove infected lower leaves to slow spread.",
            "Stake/trellis plants to improve air circulation."
        ],
        "prevention": [
            "Mulch heavily to prevent soilborne spore splash.",
            "Rotate crops — 2-3 year break from solanaceae.",
            "Use resistant varieties (Iron Lady, Jasper).",
            "Begin fungicide program at first sign or preventively during warm/humid periods."
        ],
        "organic_options": ["Copper fungicides", "Bacillus subtilis", "Neem oil", "Bicarb + oil spray"],
        "risk_factors": ["Warm days (24–29°C) with dew or light rain", "Plant stress", "Dense canopy"],
        "weather_risk": {"high_humidity": True, "rain_triggered": False, "temp_range": "24-29°C"}
    },
    "Tomato___Late_blight": {
        "common_name": "Tomato Late Blight",
        "pathogen": "Phytophthora infestans (oomycete)",
        "severity": "critical",
        "description": "Large, greasy dark brown lesions on leaves/stems; white sporulation on leaf undersides. Fruit develops firm brown rot. Can wipe out crop in days.",
        "immediate_actions": [
            "Apply systemic fungicide (mefenoxam or cymoxanil) IMMEDIATELY.",
            "Remove and bag all infected material — do NOT compost.",
            "Alert neighboring growers — spores travel miles in wind."
        ],
        "prevention": [
            "Monitor weather-based late blight forecasts.",
            "Apply preventive contact fungicides weekly during high-risk weather.",
            "Plant resistant varieties (Mountain Merit, Defiant).",
            "Avoid overhead irrigation — use drip.",
            "Destroy cull piles and volunteer tomatoes."
        ],
        "organic_options": ["Copper fungicides (bordeaux, copper hydroxide)", "Weekly applications required for protection"],
        "risk_factors": ["Cool (10–21°C) and wet weather", "High humidity >90%", "Airborne spore dispersal"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "10-21°C"}
    },
    "Tomato___Leaf_Mold": {
        "common_name": "Tomato Leaf Mold",
        "pathogen": "Passalora fulva (fungus)",
        "severity": "moderate",
        "description": "Pale greenish-yellow spots on upper leaf surface; olive-green to brown velvety mold on undersides. Mainly a greenhouse/high-tunnel problem.",
        "immediate_actions": [
            "Immediately reduce humidity in greenhouse — improve ventilation.",
            "Apply chlorothalonil, mancozeb, or copper fungicide.",
            "Remove infected leaves."
        ],
        "prevention": [
            "Maintain relative humidity below 85%.",
            "Use overhead fans for air circulation.",
            "Plant resistant varieties (most modern hybrids have Cf resistance genes).",
            "Space plants adequately and prune for airflow."
        ],
        "organic_options": ["Copper-based sprays", "Bacillus subtilis (limited)", "Potassium bicarbonate"],
        "risk_factors": ["High humidity (>85%)", "Poor ventilation in tunnels/greenhouses", "Dense canopy"],
        "weather_risk": {"high_humidity": True, "rain_triggered": False, "temp_range": "24-26°C"}
    },
    "Tomato___Septoria_leaf_spot": {
        "common_name": "Tomato Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici (fungus)",
        "severity": "moderate",
        "description": "Numerous small circular spots with dark borders and light gray/tan centers, often with dark dots (pycnidia) visible in center. Starts on lowest leaves.",
        "immediate_actions": [
            "Apply chlorothalonil or mancozeb fungicide — most effective protectant.",
            "Remove infected lower leaves.",
            "Stake plants and mulch soil to prevent splash."
        ],
        "prevention": [
            "Crop rotation — 2 years minimum away from solanaceae.",
            "Mulch heavily to reduce spore splash from soil.",
            "Stake/trellis to keep foliage off ground.",
            "Begin fungicide program preventively in wet seasons."
        ],
        "organic_options": ["Copper fungicides", "Bacillus subtilis", "Neem oil"],
        "risk_factors": ["Warm (20–25°C) wet weather", "Soilborne overwintering inoculum", "Low-lying foliage"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "20-25°C"}
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "common_name": "Two-Spotted Spider Mite",
        "pathogen": "Tetranychus urticae (arachnid pest — not a fungus)",
        "severity": "moderate",
        "description": "Tiny yellow stippling on leaves; fine webbing on undersides. Leaves eventually turn bronze/brown and drop. Mites barely visible to naked eye.",
        "immediate_actions": [
            "Apply miticide (abamectin, bifenazate, or spiromesifen) — rotate modes of action.",
            "Spray leaf undersides thoroughly — that's where mites live.",
            "Release predatory mites (Phytoseiulus persimilis) in greenhouse settings."
        ],
        "prevention": [
            "Avoid plant water stress — stressed plants are far more susceptible.",
            "Avoid broad-spectrum insecticides that kill beneficial predatory mites.",
            "Maintain relative humidity above 50% — mites thrive in hot dry conditions.",
            "Scout leaf undersides regularly in hot dry weather."
        ],
        "organic_options": ["Insecticidal soap (spray undersides)", "Neem oil", "Predatory mites (biocontrol)", "Strong water spray to dislodge mites"],
        "risk_factors": ["Hot dry weather (>30°C)", "Drought stress", "Dusty conditions suppressing natural enemies"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": ">30°C"}
    },
    "Tomato___Target_Spot": {
        "common_name": "Tomato Target Spot",
        "pathogen": "Corynespora cassiicola (fungus)",
        "severity": "moderate",
        "description": "Brown lesions with concentric rings on leaves, stems, and fruit. Can cause significant defoliation and fruit rot in warm humid conditions.",
        "immediate_actions": [
            "Apply azoxystrobin, chlorothalonil, or fluxapyroxad fungicide.",
            "Remove infected leaves.",
            "Improve canopy ventilation."
        ],
        "prevention": [
            "Stake and prune for good airflow.",
            "Avoid overhead irrigation.",
            "Rotate crops away from solanaceae.",
            "Use preventive fungicide program in warm humid regions."
        ],
        "organic_options": ["Copper fungicides", "Bacillus subtilis"],
        "risk_factors": ["Warm (24–32°C) humid conditions", "Dense canopy", "Overhead irrigation"],
        "weather_risk": {"high_humidity": True, "rain_triggered": True, "temp_range": "24-32°C"}
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "common_name": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Begomovirus (virus) — transmitted by Bemisia tabaci whitefly",
        "severity": "critical",
        "description": "Upward leaf curling and yellowing of leaf margins. Severe stunting. Flowers drop without setting fruit. No cure — infected plants must be removed.",
        "immediate_actions": [
            "Immediately remove and destroy all infected plants — bag before moving.",
            "Apply systemic insecticide (imidacloprid soil drench) to remaining plants to kill whitefly vector.",
            "Install fine mesh insect screens on greenhouse openings.",
            "Apply reflective silver mulch — repels whiteflies."
        ],
        "prevention": [
            "Plant TYLCV-resistant varieties (Tygress, Shanty, Hazera lines).",
            "Establish whitefly monitoring with yellow sticky traps from transplanting.",
            "Use transplants from certified TYLCV-free sources.",
            "Maintain strict whitefly management program throughout season."
        ],
        "organic_options": ["Reflective mulch", "Insecticidal soap for whitefly control", "Neem oil", "Kaolin clay"],
        "risk_factors": ["Presence of B. tabaci whitefly", "Warm climates year-round", "Proximity to other infected crops"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },
    "Tomato___Tomato_mosaic_virus": {
        "common_name": "Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tobamovirus (virus) — mechanically transmitted",
        "severity": "high",
        "description": "Mosaic pattern of light/dark green on leaves; leaves may be distorted or fernleaf. Stunted growth. Fruit may show internal browning. Extremely stable — survives in soil and on tools for years.",
        "immediate_actions": [
            "Remove and destroy infected plants immediately.",
            "Disinfect ALL tools with 10% bleach or 70% alcohol — ToMV is extremely persistent.",
            "Wash hands thoroughly before handling plants.",
            "Stop smoking near plants (tobacco is a host — smokers can transmit virus)."
        ],
        "prevention": [
            "Use ToMV-resistant varieties (resistance gene Tm-2²).",
            "Use certified virus-free seed — or treat seed with 10% trisodium phosphate.",
            "Disinfect greenhouse structures and equipment between crops.",
            "Control aphids and other sucking insects (though ToMV is mainly mechanical)."
        ],
        "organic_options": ["No curative options", "Prevention through sanitation is the only strategy", "Milk spray (diluted) may have limited virus-inactivating properties"],
        "risk_factors": ["Contaminated tools", "Infected transplants", "Smokers handling plants"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    },
    "Tomato___healthy": {
        "common_name": "Healthy Tomato",
        "pathogen": None,
        "severity": "none",
        "description": "Plant appears healthy with no visible disease symptoms.",
        "immediate_actions": ["No treatment required. Continue current management."],
        "prevention": [
            "Scout plants twice weekly during vegetative growth, daily during fruiting.",
            "Maintain consistent soil moisture to prevent blossom end rot.",
            "Begin preventive fungicide program if wet weather forecast."
        ],
        "organic_options": [],
        "risk_factors": ["Late blight risk spikes in cool wet weather", "Bacterial spot after storms"],
        "weather_risk": {"high_humidity": False, "rain_triggered": False, "temp_range": "any"}
    }
}

# Severity levels for UI color coding
SEVERITY_COLORS = {
    "none": "#22c55e",      # green
    "low": "#84cc16",       # lime
    "moderate": "#f59e0b",  # amber
    "high": "#ef4444",      # red
    "critical": "#7c3aed"   # purple
}

def get_remedy(class_name: str) -> dict:
    """Get remedy info for a class name, with fallback."""
    return REMEDY_DB.get(class_name, {
        "common_name": class_name.replace("___", " ").replace("_", " "),
        "pathogen": "Unknown",
        "severity": "moderate",
        "description": "Disease detected. Consult local agricultural extension for specific guidance.",
        "immediate_actions": ["Consult local agricultural extension officer.", "Apply broad-spectrum fungicide as precaution.", "Isolate affected plants."],
        "prevention": ["Maintain good field sanitation", "Ensure proper spacing for airflow", "Use certified disease-free planting material"],
        "organic_options": ["Neem oil spray", "Copper-based fungicides"],
        "risk_factors": ["High humidity", "Poor air circulation"],
        "weather_risk": {"high_humidity": True, "rain_triggered": False, "temp_range": "any"}
    })
