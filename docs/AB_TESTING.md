# Phase 8: A/B testing (concepts + RecoSphere me kaise bana)

Offline gate (Phase 7d) ek hi sawaal ka jawab deta hai: "kya naya model **purane logs** par kharab to nahi hai?" A/B test ka sawaal alag hai: "**asli users** ko naya model dikhane se unka behaviour badla ya nahi?" Offline metric (NDCG) aur online metric (clicks) hamesha saath nahi chalte. Isliye production me dono chahiye: pehle gate, phir A/B.

```
naya model -> offline gate (sasta, paired, same logs) -> A/B (mehenga, asli users) -> ship / don't ship
```

---

## 1. Randomization: kisko kaunsa arm milega

**Unit = user, request nahi.** Agar har request par alag arm milta, to ek hi user kabhi model A dekhta, kabhi B. Uska experience toot jaata aur uske clicks dono arms me bant jaate, jisse observations independent nahi rehte. Isliye user ek baar assign hota hai aur poore experiment me usi arm me rehta hai.

**Hashing, database nahi:**
```
bucket = sha256(salt + ":" + user_id) mod 10,000
bucket < 5000 -> control, warna treatment        (50/50)
```
- **Deterministic:** wahi user, wahi arm. Har worker par, har restart ke baad, kuch store karne ki zaroorat nahi.
- **Salt har experiment ka naya:** bina salt ke, jo users pichhle experiment me treatment me the, wahi agle me bhi treatment me aate. Tab ek experiment ka asar (novelty, ya model ki wajah se badla behaviour) agle me ghus jaata. `tests/test_experiments.py` check karta hai ki naya salt users ko ~50% shuffle karta hai.

**Control = production.** `control` arm hamesha wahi model serve karta hai jo `production.json` point karta hai. Baaki arms experiment shuru hote waqt pin ho jaate hain. **Experiment ke beech promote mat karo**, warna control badal jaayega. `/health` aisa hone par `experiment_warning` dikhata hai.

Do tarah ke experiments hote hain:
- **Model test:** `treatment: runs/<run_id>` (naya model)
- **Policy test:** `treatment: production` + `new_item_slots: 0` (same model, alag re-rank rule)

## 2. Logging: kisne kya dekha, kisne kya kiya

Analysis do logs ko jod kar hoti hai:

| Log | Kaun likhta hai | Kya |
|---|---|---|
| `exposures-<pid>.jsonl` | API (server) | request_id, user, arm, model_run, dikhaye gaye items, latency |
| `clicks-<pid>.jsonl` | `/feedback` (client) | request_id, user, item, action |

- **Exposure server likhta hai, client nahi.** Client bata sakta hai ki usne kya dikhaya, lekin agar uska logging toota ho to data chupchaap ghalat hoga. Server ka log hi source of truth hai.
- **Click ka arm server dobara nikalta hai** (hash se), client se nahi leta.
- **Forced requests** (`?force_arm=treatment`, jo QA ya debugging ke liye hoti hain) logs me `forced: true` ke saath jaati hain, aur analysis unhe hata deta hai.
- Har worker process apni alag file likhta hai, isliye locking ki zaroorat nahi.

## 3. Analysis ke steps, isi order me

`python -m ab.analyze` ye sab karta hai. Order zaroori hai, kyunki har step agle step ko invalid kar sakta hai.

**(a) Data checks.** Do cheezein 0 honi chahiye: aise clicks jinka koi exposure nahi, aur aise users jo ek se zyada arms me dikhe. Agar ye 0 nahi hain, to logging me bug hai.

**(b) SRM (Sample Ratio Mismatch).** 50/50 split tha, to user counts bhi lagbhag 50/50 hone chahiye. Chi-square test ka p < 0.001 aaye to **ruk jao**. Iska matlab kahin bug hai: kisi arm ke users crash ho rahe hain, bots filter ho rahe hain, ya logging toota hai. SRM ke saath lift ka number aaye bhi, to bekaar hai.

**(c) A/A calibration.** Control arm ke users ko randomly do hisson me baanto, 500 baar. Dono hisse same model dekh rahe hain, isliye "farak hai" sirf ~5% baar aana chahiye (alpha). Isse zyada aaye to stats ka code toota hai, jaise galat unit ya correlated observations.

**(d) Metric, per user.** Primary metric hai `clicks per user`, secondary hai `converted` (koi bhi click hua ya nahi). Lift = treatment/control − 1. Lift ka CI delta method se nikalta hai, kyunki do noisy means ka ratio hai.

**(e) CUPED.** Har user ki **experiment se pehle** ki activity (X) uske outcome (Y) ka kuch hissa predict karti hai. Us predictable hisse ko hata do:
```
Y_cuped = Y − θ (X − mean X),   θ = cov(X, Y) / var(X)
```
X treatment se pehle measure hua tha, isliye uska expected effect dono arms me same hai. Effect estimate wahi rehta hai, sirf variance `corr(X, Y)²` jitna kam ho jaata hai. **Sach:** hamare synthetic data par variance sirf ~1% kam hua, kyunki synthetic users ki pichhli activity aage ke clicks ko predict hi nahi karti (corr 0.03–0.10). Asli data par apni report ka "CUPED var. reduction" column dekhna.

**(f) Guardrails.** Agar treatment clicks badhaye lekin p95 latency 25% badh jaaye, ya errors badh jaayein, to woh jeet nahi hai. Report me dekhe jaate hain: latency p50/p95, fallback share, error share, aur kitne alag items dikhe (coverage).

**(g) Power / MDE.** Is sawaal ka jawab experiment **shuru karne se pehle** chahiye:
```
MDE = (z_{1−α/2} + z_power) · sd · sqrt(2 / n_per_arm)
```
Iska matlab: "is traffic ke saath sabse chhota kitna farak pakad sakte hain?" Agar expected improvement MDE se chhota hai, to experiment chalana hi bekaar hai. Synthetic data par 3,000 users per arm ke saath MDE ~18% aaya, aur 2% lift ke liye **~2.4 lakh users per arm** chahiye.

**(h) Decision.** Order zaroori hai:

| Situation | Decision |
|---|---|
| SRM | INVALID: bug theek karo, phir rerun |
| Guardrail fail (latency, errors, **coverage −5% se zyada**, **new-item share aadha se kam**) | DON'T SHIP |
| MDE > 10% (`--max-mde`) | **NO DECISION (underpowered)**: kitne users chahiye, ye batata hai |
| CUPED lift CI > 0 | SHIP |
| CI < 0 | DON'T SHIP |
| CI 0 cross kare | INCONCLUSIVE |

**Underpowered rule kyun:** pehle asli-data run me sirf 5,111 users the (MDE 39.5%), aur "+35% clicks, p = 0.02" dekh ke analyzer ne SHIP bol diya. Jab power itni kam ho, to jo result "significant" aata hai woh aksar asli effect se kaafi bada hota hai. Isko **type-M error** (ya winner's curse) kehte hain. Same users par interleaving ne koi farak nahi dikhaya (p = 0.72).

**Coverage/new-item guardrails kyun:** naye-games ke slots isliye hain ki catalog coverage aur naye games ko exposure mile. Unhe hatane se short-term clicks badh sakte hain, lekin distinct items 9% gire. Pehle ye rule check hi nahi hota tha.

## 4. Peeking: roz p-value dekhna kyun galat hai

Har din p-value check karna aur jis din p < 0.05 aaye usi din "jeet gaye!" bol dena: ye 20 baar sikka uchhaalne jaisa hai, kabhi na kabhi lucky aayega. Hamare A/A simulation me 20 baar dekhne par false-positive rate **22–28%** aaya, jabki end me ek baar dekhne par ~5%.

Fix ke do tareeke hain:
- Sample size pehle fix karo (section g) aur end me ek hi baar dekho; **ya**
- sequential test use karo (mSPRT / always-valid p-values), jo baar-baar dekhne ke liye hi bana hai.

Grafana me live clicks-per-arm panel sirf **monitoring** ke liye hai (kuch toota to nahi), faisla lene ke liye nahi.

## 5. Interleaving: kam users me ranking compare karna

Ek hi user ko **ek mili hui list** dikhao, jisme dono rankers ke items hon (team-draft: dono baari-baari apna top item daalte hain). Jis team ke items par zyada click aaye, woh jeeti. Same user aur same visit hone se users ke beech ka farak cancel ho jaata hai. Literature (Chapelle et al. 2012) ke hisaab se ye A/B se 10–100x kam users me jawab de deta hai.

**Hamare simulation me ye fayda nahi dikha** (power table me A/B aur interleaving lagbhag barabar aaye). Wajah: hamare click model me noise zyada tar ek user ke andar ka hai, ki kaunsa item dekha aur kaunsa click hua. Users ke beech ka farak kam hai. Ye simulation ki limit hai, interleaving ki nahi.

Interleaving ki ek aur limit: ye sirf batata hai ki "kaunsi ranking behtar lagi". Revenue ya retention jaise business metrics ka jawab nahi deta. Isliye ise A/B se pehle ek fast filter ki tarah use karte hain.

## 6. Hamare paas asli users nahi hain: simulation

`ab/simulate.py` har user ko uske **asli future** (test window ke interactions) ke saath replay karta hai:
```
P(user position p dekhe)  = 1 / p^0.7                       (neeche wale items kam dekhe jaate hain)
P(click | dekha)          = 0.6  agar item user ke asli future me hai, warna 0.01
```

- **Asli hai:** API, assignment, models, exposure aur click logs, saare stats.
- **Simulated hai:** clicks.
- **Limits:** dikhane se interest paida hona (counterfactual effect) model me nahi hai. Har user ek hi baar aata hai. Novelty effect bhi nahi hai.

Resume ya interview me ise **"simulated A/B framework"** kehna, aur lift ke numbers kabhi production results ki tarah mat batana.

## 7. Synthetic data par chale do experiments

| Experiment | Lift (clicks/user) | 95% CI | Decision |
|---|---|---|---|
| `bad1` (2 trees, 1 epoch) vs production | −33% | [−42%, −23%] | DON'T SHIP |
| seed2 vs production (seed1) | +2% | [−11%, +15%] | INCONCLUSIVE |

**Asli data (91,381 users), policy test "2 naye-game slots vs 0":**
- Clicks: −0.8% [−6.9%, +5.4%]. Yaani koi farak nahi.
- New-item share 12.4% se 2.6% par gira, aur coverage −6% hui. Dono guardrails FAIL → **slots rakho**.
- Same experiment ke sirf 5,111 users par "+35%, p = 0.02" aaya tha. Ye type-M error tha, aur poore data par effect gayab ho gaya.

Doosri row ka sabak: **same do models** ke liye offline paired gate ka CI ~±2% tha, aur online A/B ka ±13%. Offline evaluation sasta aur zyada sensitive hai. Online A/B wahi pakadta hai jo offline nahi pakad sakta (asli behaviour, latency, guardrails), lekin uski keemat hai bahut zyada users.

## 8. Commands

```powershell
# 1. experiment config
copy experiments\example.yaml experiments\active.yaml      # treatment run id edit karo, salt badlo
# 2. API (EXPOSURE_DIR set hona chahiye)
$env:EXPOSURE_DIR="ab_logs"; $env:EXPERIMENT_CONFIG="experiments/active.yaml"
uvicorn serve.app:app --host 127.0.0.1 --port 8000
# 3. simulated traffic (+ interleaving)
python -m ab.simulate --api http://127.0.0.1:8000 --n-users 20000 --interleave --out ab_runs/interleave.jsonl
# 4. report
python -m ab.analyze --logs ab_logs --experiment experiments/active.yaml --interleave ab_runs/interleave.jsonl
# 5. experiment band: active.yaml delete karo (API 10 s me control par wapas aa jaati hai)
```

## Interview me kaise batana hai

"Maine ek A/B framework banaya. Users ko salted hash se assign kiya, server-side exposure log rakha aur client-side clicks ko request_id se joda. Analysis me pehle SRM aur A/A calibration chalaye, phir CUPED, guardrails aur power analysis. Users asli nahi the, isliye asli future interactions par position-biased click model se traffic simulate kiya. Sabse bada sabak ye tha ki same models par offline paired gate ka CI ~±2% tha aur A/B ka ±13%. Isliye production me offline gate pehle chalta hai, aur A/B sirf unhi candidates par jo gate pass kar chuke hon."
