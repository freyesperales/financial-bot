"""
Universo de Acciones y ETFs - Compatible con N26
~160 símbolos: acciones diversificadas + ETFs disponibles en N26 (NYSE / NASDAQ)

Nota sobre ETFs en N26:
N26 permite comprar ETFs listados en NYSE/NASDAQ mediante su función de inversión.
Los ETFs aquí incluidos son de alta liquidez y ampliamente disponibles.
"""

# ─────────────────────────────────────────────
#  ACCIONES POR SECTOR
# ─────────────────────────────────────────────

STOCK_UNIVERSE = {
    # Tecnología (22)
    "Technology": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet (Google)
        "NVDA",   # NVIDIA
        "META",   # Meta (Facebook)
        "TSLA",   # Tesla
        "AMD",    # AMD
        "CRM",    # Salesforce
        "ADBE",   # Adobe
        "INTC",   # Intel
        "ORCL",   # Oracle
        "CSCO",   # Cisco
        "QCOM",   # Qualcomm
        "TXN",    # Texas Instruments
        "AVGO",   # Broadcom
        "MU",     # Micron Technology
        "AMAT",   # Applied Materials
        "NOW",    # ServiceNow
        "SNOW",   # Snowflake
        "PANW",   # Palo Alto Networks
        "CRWD",   # CrowdStrike
        "NET",    # Cloudflare
    ],

    # Financiero (15)
    "Financials": [
        "JPM",    # JPMorgan Chase
        "BAC",    # Bank of America
        "WFC",    # Wells Fargo
        "GS",     # Goldman Sachs
        "MS",     # Morgan Stanley
        "V",      # Visa
        "MA",     # Mastercard
        "AXP",    # American Express
        "BRK-B",  # Berkshire Hathaway B
        "C",      # Citigroup
        "BLK",    # BlackRock
        "SCHW",   # Charles Schwab
        "COF",    # Capital One
        "USB",    # US Bancorp
        "PGR",    # Progressive Insurance
    ],

    # Salud (15)
    "Healthcare": [
        "JNJ",    # Johnson & Johnson
        "UNH",    # UnitedHealth
        "PFE",    # Pfizer
        "ABBV",   # AbbVie
        "TMO",    # Thermo Fisher
        "MRK",    # Merck
        "LLY",    # Eli Lilly
        "ABT",    # Abbott
        "BMY",    # Bristol-Myers Squibb
        "AMGN",   # Amgen
        "GILD",   # Gilead Sciences
        "ISRG",   # Intuitive Surgical
        "VRTX",   # Vertex Pharmaceuticals
        "REGN",   # Regeneron
        "MDT",    # Medtronic
    ],

    # Consumo Discrecional (12)
    "Consumer Discretionary": [
        "AMZN",   # Amazon
        "HD",     # Home Depot
        "MCD",    # McDonald's
        "NKE",    # Nike
        "SBUX",   # Starbucks
        "TGT",    # Target
        "COST",   # Costco
        "LOW",    # Lowe's
        "TJX",    # TJX Companies
        "BKNG",   # Booking Holdings
        "GM",     # General Motors
        "F",      # Ford
    ],

    # Consumo Básico (8)
    "Consumer Staples": [
        "WMT",    # Walmart
        "PG",     # Procter & Gamble
        "KO",     # Coca-Cola
        "PEP",    # PepsiCo
        "PM",     # Philip Morris
        "MDLZ",   # Mondelez
        "CL",     # Colgate-Palmolive
        "KHC",    # Kraft Heinz
    ],

    # Energía — Petróleo, Gas & Uranio (15)
    "Energy": [
        # Petróleo & Gas
        "XOM",    # Exxon Mobil
        "CVX",    # Chevron
        "COP",    # ConocoPhillips
        "SLB",    # Schlumberger
        "EOG",    # EOG Resources
        "OXY",    # Occidental Petroleum
        "PSX",    # Phillips 66
        "VLO",    # Valero Energy
        "MPC",    # Marathon Petroleum
        "HAL",    # Halliburton
        # Uranio & Energía Nuclear (alta demanda por IA y electrificación)
        "UEC",    # Uranium Energy Corp
        "CCJ",    # Cameco Corp (mayor productor mundial de uranio)
        "NXE",    # NexGen Energy
        "UUUU",   # Energy Fuels Inc.
        "DNN",    # Denison Mines
    ],

    # Comunicaciones (8)
    "Communications": [
        "DIS",    # Disney
        "NFLX",   # Netflix
        "CMCSA",  # Comcast
        "T",      # AT&T
        "VZ",     # Verizon
        "SPOT",   # Spotify
        "WBD",    # Warner Bros Discovery
        "EA",     # Electronic Arts
    ],

    # Industrial (12)
    "Industrials": [
        "BA",     # Boeing
        "CAT",    # Caterpillar
        "GE",     # General Electric
        "UPS",    # UPS
        "HON",    # Honeywell
        "RTX",    # Raytheon Technologies
        "LMT",    # Lockheed Martin
        "DE",     # John Deere
        "MMM",    # 3M
        "FDX",    # FedEx
        "WM",     # Waste Management
        "ETN",    # Eaton
    ],

    # Materiales (5)
    "Materials": [
        "LIN",    # Linde
        "APD",    # Air Products
        "NEM",    # Newmont Mining
        "FCX",    # Freeport-McMoRan (Cobre)
        "NUE",    # Nucor (Acero)
    ],

    # Real Estate / REITs (5)
    "Real Estate": [
        "AMT",    # American Tower
        "PLD",    # Prologis
        "EQIX",   # Equinix
        "SPG",    # Simon Property Group
        "O",      # Realty Income
    ],

    # Utilities (5)
    "Utilities": [
        "NEE",    # NextEra Energy
        "DUK",    # Duke Energy
        "SO",     # Southern Company
        "D",      # Dominion Energy
        "AEP",    # American Electric Power
    ],
}

# ─────────────────────────────────────────────
#  ETFs POR CATEGORÍA
# ─────────────────────────────────────────────

ETF_UNIVERSE = {
    # ETFs de Índices Principales (8)
    # Comprar "el mercado entero" con un solo instrumento
    "ETF - Indices": [
        "SPY",    # SPDR S&P 500 ETF — el más negociado del mundo
        "QQQ",    # Invesco NASDAQ-100 — top 100 tecnológicas
        "IWM",    # iShares Russell 2000 — small caps EEUU
        "DIA",    # SPDR Dow Jones Industrial Average
        "MDY",    # SPDR S&P MidCap 400
        "VTI",    # Vanguard Total Stock Market (todo EEUU)
        "EFA",    # iShares MSCI EAFE — mercados desarrollados ex-EEUU
        "EEM",    # iShares MSCI Emerging Markets — economías emergentes
    ],

    # ETFs Sectoriales (8)
    # Apostar por un sector sin elegir empresa individual
    "ETF - Sectorial": [
        "XLK",    # Technology Select Sector SPDR
        "XLF",    # Financials Select Sector SPDR
        "XLE",    # Energy Select Sector SPDR
        "XLV",    # Health Care Select Sector SPDR
        "XLY",    # Consumer Discretionary Select Sector SPDR
        "XLI",    # Industrials Select Sector SPDR
        "SOXX",   # iShares Semiconductor ETF
        "SMH",    # VanEck Semiconductor ETF
    ],

    # ETFs Temáticos / Innovación (11)
    # Tendencias de largo plazo: energía limpia, IA, uranio, litio...
    "ETF - Tematico": [
        "ARKK",   # ARK Innovation ETF (disrupción tecnológica)
        "ARKG",   # ARK Genomic Revolution (biotech & genómica)
        "BOTZ",   # Global X Robotics & AI ETF
        "ICLN",   # iShares Global Clean Energy ETF
        "LIT",    # Global X Lithium & Battery Tech ETF
        "URA",    # Global X Uranium ETF ← sector nuclear/uranio
        "URNM",   # Sprott Uranium Miners ETF ← mineras de uranio
        "REMX",   # VanEck Rare Earth & Strategic Metals ETF
        "HACK",   # ETFMG Prime Cyber Security ETF
        "JETS",   # US Global Jets ETF (aerolíneas)
        "BLOK",   # Amplify Transformational Data Sharing (blockchain)
    ],

    # ETFs de Materias Primas (6)
    # Oro, plata, petróleo — cobertura contra inflación y volatilidad
    "ETF - Commodities": [
        "GLD",    # SPDR Gold Shares — oro físico
        "SLV",    # iShares Silver Trust — plata física
        "GDX",    # VanEck Gold Miners ETF — empresas mineras de oro
        "GDXJ",   # VanEck Junior Gold Miners ETF — mineras pequeñas
        "USO",    # United States Oil Fund — precio del petróleo WTI
        "PDBC",   # Invesco Diversified Commodity Strategy — cesta diversificada
    ],

    # ETFs de Renta Fija (4)
    # Bonos — menor riesgo, flujo de ingresos, diversificación
    "ETF - Bonos": [
        "TLT",    # iShares 20+ Year Treasury Bond — bonos EEUU largo plazo
        "AGG",    # iShares Core US Aggregate Bond — mercado de bonos amplio
        "HYG",    # iShares iBoxx High Yield Corporate Bond — bonos corporativos
        "TIP",    # iShares TIPS Bond — bonos ligados a inflación
    ],
}

# ─────────────────────────────────────────────
#  INFORMACIÓN: NOMBRE COMPLETO + ISIN
# ─────────────────────────────────────────────
# Nota: verificar ISIN en N26 o broker antes de operar.
# Las empresas con sede fuera de EE.UU. tienen ISIN con prefijo distinto a "US".

STOCK_INFO = {
    # ── Technology ──────────────────────────────────────────────────────────
    "AAPL":  {"name": "Apple Inc.",                       "isin": "US0378331005"},
    "MSFT":  {"name": "Microsoft Corp.",                  "isin": "US5949181045"},
    "GOOGL": {"name": "Alphabet Inc. (Google) Cl.A",      "isin": "US02079K3059"},
    "NVDA":  {"name": "NVIDIA Corp.",                     "isin": "US67066G1040"},
    "META":  {"name": "Meta Platforms Inc.",               "isin": "US30303M1027"},
    "TSLA":  {"name": "Tesla Inc.",                       "isin": "US88160R1014"},
    "AMD":   {"name": "Advanced Micro Devices Inc.",      "isin": "US0079031078"},
    "CRM":   {"name": "Salesforce Inc.",                  "isin": "US79466L3024"},
    "ADBE":  {"name": "Adobe Inc.",                       "isin": "US00724F1012"},
    "INTC":  {"name": "Intel Corp.",                      "isin": "US4581401001"},
    "ORCL":  {"name": "Oracle Corp.",                     "isin": "US68389X1054"},
    "CSCO":  {"name": "Cisco Systems Inc.",               "isin": "US17275R1023"},
    "QCOM":  {"name": "Qualcomm Inc.",                    "isin": "US7475251036"},
    "TXN":   {"name": "Texas Instruments Inc.",           "isin": "US8825081040"},
    "AVGO":  {"name": "Broadcom Inc.",                    "isin": "US11135F1012"},
    "MU":    {"name": "Micron Technology Inc.",           "isin": "US5951121038"},
    "AMAT":  {"name": "Applied Materials Inc.",           "isin": "US0382221051"},
    "NOW":   {"name": "ServiceNow Inc.",                  "isin": "US81762P1021"},
    "SNOW":  {"name": "Snowflake Inc.",                   "isin": "US8334451098"},
    "PANW":  {"name": "Palo Alto Networks Inc.",          "isin": "US6974351057"},
    "CRWD":  {"name": "CrowdStrike Holdings Inc.",        "isin": "US22788C1053"},
    "NET":   {"name": "Cloudflare Inc.",                  "isin": "US18915M1071"},
    # ── Financials ──────────────────────────────────────────────────────────
    "JPM":   {"name": "JPMorgan Chase & Co.",             "isin": "US46625H1005"},
    "BAC":   {"name": "Bank of America Corp.",            "isin": "US0605051046"},
    "WFC":   {"name": "Wells Fargo & Co.",                "isin": "US9497461015"},
    "GS":    {"name": "Goldman Sachs Group Inc.",         "isin": "US38141G1040"},
    "MS":    {"name": "Morgan Stanley",                   "isin": "US6174464486"},
    "V":     {"name": "Visa Inc.",                        "isin": "US92826C8394"},
    "MA":    {"name": "Mastercard Inc.",                  "isin": "US57636Q1040"},
    "AXP":   {"name": "American Express Co.",             "isin": "US0258161092"},
    "BRK-B": {"name": "Berkshire Hathaway Inc. Cl.B",    "isin": "US0846707026"},
    "C":     {"name": "Citigroup Inc.",                   "isin": "US1729674242"},
    "BLK":   {"name": "BlackRock Inc.",                   "isin": "US09247X1019"},
    "SCHW":  {"name": "Charles Schwab Corp.",             "isin": "US8085131055"},
    "COF":   {"name": "Capital One Financial Corp.",      "isin": "US14040H1059"},
    "USB":   {"name": "U.S. Bancorp",                     "isin": "US9029733048"},
    "PGR":   {"name": "Progressive Corp.",                "isin": "US7433151039"},
    # ── Healthcare ──────────────────────────────────────────────────────────
    "JNJ":   {"name": "Johnson & Johnson",                "isin": "US4781601046"},
    "UNH":   {"name": "UnitedHealth Group Inc.",          "isin": "US91324P1021"},
    "PFE":   {"name": "Pfizer Inc.",                      "isin": "US7170811035"},
    "ABBV":  {"name": "AbbVie Inc.",                      "isin": "US00287Y1091"},
    "TMO":   {"name": "Thermo Fisher Scientific Inc.",    "isin": "US8835561023"},
    "MRK":   {"name": "Merck & Co. Inc.",                 "isin": "US58933Y1055"},
    "LLY":   {"name": "Eli Lilly and Co.",                "isin": "US5324571083"},
    "ABT":   {"name": "Abbott Laboratories",              "isin": "US0028241000"},
    "BMY":   {"name": "Bristol-Myers Squibb Co.",         "isin": "US1101221083"},
    "AMGN":  {"name": "Amgen Inc.",                       "isin": "US0311621009"},
    "GILD":  {"name": "Gilead Sciences Inc.",             "isin": "US3755581036"},
    "ISRG":  {"name": "Intuitive Surgical Inc.",          "isin": "US46120E6023"},
    "VRTX":  {"name": "Vertex Pharmaceuticals Inc.",      "isin": "US92532F1003"},
    "REGN":  {"name": "Regeneron Pharmaceuticals Inc.",   "isin": "US75886F1075"},
    "MDT":   {"name": "Medtronic plc",                    "isin": "IE00BTN1Y115"},   # Irlanda
    # ── Consumer Discretionary ──────────────────────────────────────────────
    "AMZN":  {"name": "Amazon.com Inc.",                  "isin": "US0231351067"},
    "HD":    {"name": "Home Depot Inc.",                  "isin": "US4370761029"},
    "MCD":   {"name": "McDonald's Corp.",                 "isin": "US5801351017"},
    "NKE":   {"name": "Nike Inc.",                        "isin": "US6541061031"},
    "SBUX":  {"name": "Starbucks Corp.",                  "isin": "US8552441094"},
    "TGT":   {"name": "Target Corp.",                     "isin": "US87612E1064"},
    "COST":  {"name": "Costco Wholesale Corp.",           "isin": "US22160K1051"},
    "LOW":   {"name": "Lowe's Companies Inc.",            "isin": "US5486611073"},
    "TJX":   {"name": "TJX Companies Inc.",               "isin": "US8725401090"},
    "BKNG":  {"name": "Booking Holdings Inc.",            "isin": "US09857L1089"},
    "GM":    {"name": "General Motors Co.",               "isin": "US37045V1008"},
    "F":     {"name": "Ford Motor Co.",                   "isin": "US3453708600"},
    # ── Consumer Staples ────────────────────────────────────────────────────
    "WMT":   {"name": "Walmart Inc.",                     "isin": "US9311421039"},
    "PG":    {"name": "Procter & Gamble Co.",             "isin": "US7427181091"},
    "KO":    {"name": "Coca-Cola Co.",                    "isin": "US1912161007"},
    "PEP":   {"name": "PepsiCo Inc.",                     "isin": "US7134481081"},
    "PM":    {"name": "Philip Morris International Inc.", "isin": "US7181721090"},
    "MDLZ":  {"name": "Mondelez International Inc.",      "isin": "US6092071058"},
    "CL":    {"name": "Colgate-Palmolive Co.",            "isin": "US1941621039"},
    "KHC":   {"name": "Kraft Heinz Co.",                  "isin": "US5007541064"},
    # ── Energy ──────────────────────────────────────────────────────────────
    "XOM":   {"name": "Exxon Mobil Corp.",                "isin": "US30231G1022"},
    "CVX":   {"name": "Chevron Corp.",                    "isin": "US1667641005"},
    "COP":   {"name": "ConocoPhillips",                   "isin": "US20825C1045"},
    "SLB":   {"name": "SLB (Schlumberger N.V.)",          "isin": "AN8068571086"},   # Curazao
    "EOG":   {"name": "EOG Resources Inc.",               "isin": "US26875P1012"},
    "OXY":   {"name": "Occidental Petroleum Corp.",       "isin": "US6745991058"},
    "PSX":   {"name": "Phillips 66",                      "isin": "US7185461040"},
    "VLO":   {"name": "Valero Energy Corp.",              "isin": "US91913Y1001"},
    "MPC":   {"name": "Marathon Petroleum Corp.",         "isin": "US56585A1025"},
    "HAL":   {"name": "Halliburton Co.",                  "isin": "US4062161017"},
    "UEC":   {"name": "Uranium Energy Corp.",             "isin": "US9171961063"},
    "CCJ":   {"name": "Cameco Corp.",                     "isin": "CA13321L1085"},   # Canadá
    "NXE":   {"name": "NexGen Energy Ltd.",               "isin": "CA65340P1062"},   # Canadá
    "UUUU":  {"name": "Energy Fuels Inc.",                "isin": "US29258N1073"},
    "DNN":   {"name": "Denison Mines Corp.",              "isin": "CA2488341090"},   # Canadá
    # ── Communications ──────────────────────────────────────────────────────
    "DIS":   {"name": "Walt Disney Co.",                  "isin": "US2546871060"},
    "NFLX":  {"name": "Netflix Inc.",                     "isin": "US64110L1061"},
    "CMCSA": {"name": "Comcast Corp.",                    "isin": "US20030N1019"},
    "T":     {"name": "AT&T Inc.",                        "isin": "US00206R1023"},
    "VZ":    {"name": "Verizon Communications Inc.",      "isin": "US92343V1044"},
    "SPOT":  {"name": "Spotify Technology S.A.",          "isin": "LU1778762911"},   # Luxemburgo
    "WBD":   {"name": "Warner Bros. Discovery Inc.",      "isin": "US9344231041"},
    "EA":    {"name": "Electronic Arts Inc.",             "isin": "US2855121099"},
    # ── Industrials ─────────────────────────────────────────────────────────
    "BA":    {"name": "Boeing Co.",                       "isin": "US0970231058"},
    "CAT":   {"name": "Caterpillar Inc.",                 "isin": "US1491231015"},
    "GE":    {"name": "GE Aerospace",                     "isin": "US3696043013"},
    "UPS":   {"name": "United Parcel Service Inc.",       "isin": "US9113121068"},
    "HON":   {"name": "Honeywell International Inc.",     "isin": "US4385161066"},
    "RTX":   {"name": "RTX Corp.",                        "isin": "US75513E1010"},
    "LMT":   {"name": "Lockheed Martin Corp.",            "isin": "US5398301094"},
    "DE":    {"name": "Deere & Co.",                      "isin": "US2441991054"},
    "MMM":   {"name": "3M Co.",                           "isin": "US88579Y1010"},
    "FDX":   {"name": "FedEx Corp.",                      "isin": "US31428X1063"},
    "WM":    {"name": "Waste Management Inc.",            "isin": "US94106L1098"},
    "ETN":   {"name": "Eaton Corporation plc",            "isin": "IE00B8KQN827"},   # Irlanda
    # ── Materials ───────────────────────────────────────────────────────────
    "LIN":   {"name": "Linde plc",                        "isin": "IE00BZ12WP82"},   # Irlanda
    "APD":   {"name": "Air Products and Chemicals Inc.",  "isin": "US0091581068"},
    "NEM":   {"name": "Newmont Corp.",                    "isin": "US6516391066"},
    "FCX":   {"name": "Freeport-McMoRan Inc.",            "isin": "US35671D8570"},
    "NUE":   {"name": "Nucor Corp.",                      "isin": "US6703461052"},
    # ── Real Estate ─────────────────────────────────────────────────────────
    "AMT":   {"name": "American Tower Corp.",             "isin": "US03027X1000"},
    "PLD":   {"name": "Prologis Inc.",                    "isin": "US74340W1036"},
    "EQIX":  {"name": "Equinix Inc.",                     "isin": "US29444U7000"},
    "SPG":   {"name": "Simon Property Group Inc.",        "isin": "US8288061091"},
    "O":     {"name": "Realty Income Corp.",              "isin": "US7561091049"},
    # ── Utilities ───────────────────────────────────────────────────────────
    "NEE":   {"name": "NextEra Energy Inc.",              "isin": "US65339F1012"},
    "DUK":   {"name": "Duke Energy Corp.",                "isin": "US26441C2044"},
    "SO":    {"name": "Southern Company",                 "isin": "US8425871071"},
    "D":     {"name": "Dominion Energy Inc.",             "isin": "US25746U1097"},
    "AEP":   {"name": "American Electric Power Co.",      "isin": "US0255371081"},
    # ── ETF - Indices ────────────────────────────────────────────────────────
    "SPY":   {"name": "SPDR S&P 500 ETF Trust",           "isin": "US78462F1030"},
    "QQQ":   {"name": "Invesco QQQ Trust (Nasdaq-100)",   "isin": "US46090E1038"},
    "IWM":   {"name": "iShares Russell 2000 ETF",         "isin": "US4642876555"},
    "DIA":   {"name": "SPDR Dow Jones Industrial ETF",    "isin": "US78467X1090"},
    "MDY":   {"name": "SPDR S&P MidCap 400 ETF",          "isin": "US78464A8707"},
    "VTI":   {"name": "Vanguard Total Stock Market ETF",  "isin": "US9229087690"},
    "EFA":   {"name": "iShares MSCI EAFE ETF",            "isin": "US4642874659"},
    "EEM":   {"name": "iShares MSCI Emerg. Markets ETF",  "isin": "US4642872349"},
    # ── ETF - Sectorial ──────────────────────────────────────────────────────
    "XLK":   {"name": "Technology Select Sector SPDR",    "isin": "US81369Y8030"},
    "XLF":   {"name": "Financials Select Sector SPDR",    "isin": "US81369Y6065"},
    "XLE":   {"name": "Energy Select Sector SPDR",        "isin": "US81369Y4052"},
    "XLV":   {"name": "Health Care Select Sector SPDR",   "isin": "US81369Y6149"},
    "XLY":   {"name": "Consumer Discr. Select Sector SPDR","isin": "US81369Y8022"},
    "XLI":   {"name": "Industrials Select Sector SPDR",   "isin": "US81369Y5521"},
    "SOXX":  {"name": "iShares Semiconductor ETF",        "isin": "US46432F8390"},
    "SMH":   {"name": "VanEck Semiconductor ETF",         "isin": "US92189F1066"},
    # ── ETF - Tematico ───────────────────────────────────────────────────────
    "ARKK":  {"name": "ARK Innovation ETF",               "isin": "US00214Q1040"},
    "ARKG":  {"name": "ARK Genomic Revolution ETF",       "isin": "US00214Q4028"},
    "BOTZ":  {"name": "Global X Robotics & AI ETF",       "isin": "US37954Y8306"},
    "ICLN":  {"name": "iShares Global Clean Energy ETF",  "isin": "US46434G8481"},
    "LIT":   {"name": "Global X Lithium & Battery Tech ETF","isin": "US37954Y6573"},
    "URA":   {"name": "Global X Uranium ETF",             "isin": "US37954Y8561"},
    "URNM":  {"name": "Sprott Uranium Miners ETF",        "isin": "US85208A1079"},
    "REMX":  {"name": "VanEck Rare Earth & Strat. Metals","isin": "US92189F3609"},
    "HACK":  {"name": "ETFMG Prime Cyber Security ETF",   "isin": "US26922A5765"},
    "JETS":  {"name": "US Global Jets ETF",               "isin": "US9035771085"},
    "BLOK":  {"name": "Amplify Transformational Data ETF","isin": "US0321081012"},
    # ── ETF - Commodities ────────────────────────────────────────────────────
    "GLD":   {"name": "SPDR Gold Shares",                 "isin": "US78463V1070"},
    "SLV":   {"name": "iShares Silver Trust",             "isin": "US46428Q1094"},
    "GDX":   {"name": "VanEck Gold Miners ETF",           "isin": "US92189F7099"},
    "GDXJ":  {"name": "VanEck Junior Gold Miners ETF",    "isin": "US92189F5093"},
    "USO":   {"name": "United States Oil Fund",           "isin": "US9120421090"},
    "PDBC":  {"name": "Invesco Optimum Yield Diversified","isin": "US46138G7495"},
    # ── ETF - Bonos ──────────────────────────────────────────────────────────
    "TLT":   {"name": "iShares 20+ Year Treasury Bond ETF","isin": "US4642874329"},
    "AGG":   {"name": "iShares Core US Aggregate Bond ETF","isin": "US4642872422"},
    "HYG":   {"name": "iShares iBoxx High Yield Corp Bond","isin": "US4642885135"},
    "TIP":   {"name": "iShares TIPS Bond ETF",            "isin": "US4642871762"},
}


def get_stock_info(symbol: str) -> dict:
    """
    Retorna {'name': '...', 'isin': '...'} para un símbolo.
    Si no está en el diccionario devuelve {'name': symbol, 'isin': 'N/D'}.
    """
    return STOCK_INFO.get(symbol, {"name": symbol, "isin": "N/D"})


# ─────────────────────────────────────────────
#  FUNCIONES DE ACCESO
# ─────────────────────────────────────────────

# Conjunto rápido para saber si un símbolo es ETF
ETF_SYMBOLS: set = {
    s for symbols in ETF_UNIVERSE.values() for s in symbols
}

def is_etf(symbol: str) -> bool:
    """Retorna True si el símbolo es un ETF"""
    return symbol in ETF_SYMBOLS

def get_all_symbols():
    """Retorna lista completa: acciones + ETFs"""
    symbols = []
    for sector_stocks in STOCK_UNIVERSE.values():
        symbols.extend(sector_stocks)
    for etf_list in ETF_UNIVERSE.values():
        symbols.extend(etf_list)
    return symbols

def get_all_stocks():
    """Retorna solo acciones (sin ETFs)"""
    symbols = []
    for sector_stocks in STOCK_UNIVERSE.values():
        symbols.extend(sector_stocks)
    return symbols

def get_all_etfs():
    """Retorna solo ETFs"""
    symbols = []
    for etf_list in ETF_UNIVERSE.values():
        symbols.extend(etf_list)
    return symbols

def get_sector(symbol):
    """Retorna sector/categoría de un símbolo"""
    for sector, stocks in STOCK_UNIVERSE.items():
        if symbol in stocks:
            return sector
    for category, etfs in ETF_UNIVERSE.items():
        if symbol in etfs:
            return category
    return "Unknown"

def get_stocks_by_sector(sector):
    """Retorna acciones de un sector"""
    return STOCK_UNIVERSE.get(sector, ETF_UNIVERSE.get(sector, []))

def get_sector_list():
    """Retorna lista de todos los sectores y categorías ETF"""
    return list(STOCK_UNIVERSE.keys()) + list(ETF_UNIVERSE.keys())


if __name__ == "__main__":
    print("=" * 70)
    print("  Universo de Inversion - Compatible con N26")
    print("=" * 70)
    print()

    print("--- ACCIONES ---")
    total_stocks = 0
    for sector, stocks in STOCK_UNIVERSE.items():
        print(f"  {sector:28} ({len(stocks):2}): {', '.join(stocks[:6])}{'...' if len(stocks)>6 else ''}")
        total_stocks += len(stocks)

    print()
    print("--- ETFs ---")
    total_etfs = 0
    for cat, etfs in ETF_UNIVERSE.items():
        print(f"  {cat:28} ({len(etfs):2}): {', '.join(etfs)}")
        total_etfs += len(etfs)

    print()
    print(f"Total acciones: {total_stocks}")
    print(f"Total ETFs    : {total_etfs}")
    print(f"Total universo: {total_stocks + total_etfs} simbolos")
    print()
    print("Todos disponibles en N26 (NYSE / NASDAQ)")
