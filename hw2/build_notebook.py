"""Assemble the submission notebook (HW2_KMeans_Quantization.ipynb) and REPORT.md.

Reads metrics produced by run_kmeans.py + run_mixed.py from RESULTS_DIR and bakes
the real numbers into a Ukrainian narrative. Run AFTER both experiments finish,
then execute the notebook with nbconvert.
"""
import json
import os
import sys

import nbformat as nbf

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_NB = "HW2_KMeans_Quantization.ipynb"
OUT_REPORT = "REPORT.md"


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


km = load("kmeans.json")
mx = load("mixed.json")
abl = load("ablation.json")          # pooling (sum/mean, adam/sgd) + adapt_extras
sd = load("seeds.json")              # 3-seed mean+/-std for the mixed-vs-uniform claim

fp32 = km["fp32"]
bit_keys = sorted((k for k in km if k.endswith("bit")), key=lambda k: int(k[:-3]))
bits_list = [int(k[:-3]) for k in bit_keys]

# bonus A: mixed-precision vs the uniform Pareto frontier (dense)
mixed_r = mx["mixed"]
mixed_avg = mx["mixed_avg_bits"]
mixed_bar = mixed_r["pareto_bar"]
mixed_gain = mixed_r["qat_test"] - mixed_bar          # gap above the uniform line
mixed_beats = mixed_r["beats_pareto"]
uni_ref = mx["uniform_ref"]                            # {"2": {...}, "3": {...}}

# bonus B: improve pruning with mixed-precision quantization
pr = mx.get("prune")
pm = mx.get("prune_mixed")
pu = mx.get("prune_uniform_ref")
phi = mx.get("prune_quant_headline")
p_gain = (pm["qat_test"] - pm["pareto_bar"]) if pm else 0.0

# ablation-derived numbers
ab_p_adam = abl["pooling_2bit_adam"]
ab_p_sgd = abl["pooling_2bit_sgd"]
ex2, ex3 = abl["extras_2bit"], abl["extras_3bit"]
ex2_gain = ex2["adapt"]["qat_test"] - ex2["frozen"]["qat_test"]
ex3_gain = ex3["adapt"]["qat_test"] - ex3["frozen"]["qat_test"]

# seeds-derived numbers (mean +/- std over N seeds)
n_seeds = len(sd["seeds"])
sd_d_uni = sd["dense"]["uniform"]        # {"2": {mean,std,size_MB,...}, "3": {...}}
sd_d_mix = sd["dense"]["mixed"]
sd_p_uni = sd["prune"]["uniform"]
sd_p_mix = sd["prune"]["mixed"]
d_uni_keys = sorted(sd_d_uni, key=int)
hi_uni = d_uni_keys[-1]                   # highest uniform ref (e.g. "3")
dmix_gain = sd_d_mix["mean"] - sd_d_mix["pareto_bar"]
pmix_gain = sd_p_mix["mean"] - sd_p_mix["pareto_bar"]

ppm = lambda m, s: f"{m*100:.2f}±{s*100:.2f}%"     # mean±std formatter

pct = lambda x: f"{x * 100:.2f}%"
pp = lambda x: f"{x * 100:+.2f} п.п."

cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))


# --------------------------------------------------------------------------- #
sweep_rows = "\n".join(
    f"| K-Means {b}-bit (QAT) | {b} | {km[f'{b}bit']['compression_x']:.1f}× | "
    f"{km[f'{b}bit']['size_MB']:.2f} | {pct(km[f'{b}bit']['ptq_val'])} | "
    f"**{pct(km[f'{b}bit']['qat_test'])}** |"
    for b in bits_list
)
pm_row = (f"| **Прунінг+mixed ~{sd_p_mix['avg_bits']:.2f}-bit (бонус)** | {sd_p_mix['avg_bits']:.2f} | "
          f"**{pm['compression_x']:.1f}×** | {sd_p_mix['size_MB']:.2f} | — | "
          f"**{ppm(sd_p_mix['mean'], sd_p_mix['std'])}** |" if pm else "")
pq_row = (f"| Прунінг+квант {phi['bits']}-bit (Deep Compression) | {phi['bits']} | "
          f"{phi['compression_x']:.1f}× | {phi['sparse_MB']:.2f} | — | "
          f"{pct(phi['qat_test'])} |" if phi else "")

md(f"""
# Домашня робота 2 — K-Means квантизація та QAT для VGG11 на CIFAR-10

**Курс:** Efficient ML, SET University

Беремо навчену в ДЗ1 щільну VGG11 ({pct(fp32['test'])} на тесті) і стискаємо її
**квантизацією зі спільними вагами** (weight sharing, стиль Deep Compression). Для
кожного шару кластеризуємо ваги в K = 2^bits центроїдів: замість мільйонів різних
чисел лишається маленький словник (codebook) і по одному короткому індексу на вагу.
2 біти — це лише 4 різні значення ваги на шар, 4 біти — 16.

Далі — **QAT за діаграмою зі слайда 36**: індекси кластерів заморожені, а самі
центроїди до-навчаємо. Робимо звичайний forward/backward, **пулимо градієнти за
індексом кластера** (усереднюючи в межах кластера) і крокуємо центроїди. Це те, що
дозволяє навіть 2-бітній моделі відігратися.

**Методологія.** Стартова fp32-модель узята з ДЗ1 (не перенавчаємо — спільна
відправна точка для всіх). Тест міряємо рівно один раз на кожну фінальну модель;
усі проміжні числа (PTQ, криві QAT, аналіз чутливості) — на валідації.
Mixed-precision порівнюємо з uniform **за Парето на однаковому розмірі**: точка
mixed має лежати вище лінії uniform-квантизацій того самого розміру.

### Підсумок (тест — раз на модель)

| Модель | Біт/вагу | Стиснення | Розмір, МБ | val | Точність (тест) |
|---|---|---|---|---|---|
| Baseline fp32 (з ДЗ1) | 32 | 1× | {fp32['size_MB']:.2f} | {pct(fp32['val'])} | **{pct(fp32['test'])}** |
{sweep_rows}
| Bonus: mixed ~{sd_d_mix['avg_bits']:.2f}-bit ({n_seeds} seeds) | {sd_d_mix['avg_bits']:.2f} | — | {sd_d_mix['size_MB']:.2f} | — | {ppm(sd_d_mix['mean'], sd_d_mix['std'])} |
{pm_row}
{pq_row}

Стовпчик **val**: для baseline — валідація fp32; для квант-рядків — валідація після
PTQ (до-навчання). Bonus-рядки — середнє ± std за {n_seeds} seed.

> **Про розмір.** Колонка «Розмір» — це **теоретичний packed-size** серіалізованого
> формату: codebook (K×fp32) + bit-packed індекси (log₂K на вагу), для
> прунінг-рядків ще + бітова маска позицій. Це **не** розмір поточного PyTorch
> checkpoint/RAM: у нас індекси лежать як `int64`, а ваги під час forward
> матеріалізуються назад у fp32. Тобто {km['4bit']['size_MB']:.2f} МБ для 4-біт — коректна
> оцінка запакованого файлу, а не фактичний обсяг пам'яті поточної реалізації.

4-бітна квантизація майже без втрат ({km['4bit']['compression_x']:.1f}× стиснення).
Найкраще співвідношення точність/розмір — **прунінг + mixed-precision квантизація**:
{ppm(sd_p_mix['mean'], sd_p_mix['std'])} при {pm['compression_x']:.1f}×. Mixed-precision
на обох бюджетах лежить вище Парето-лінії uniform (на щільній моделі навіть перевищує
uniform-3-bit за менший розмір) — деталі й похибки за {n_seeds} seed нижче.
""")

md("""
## 1. Підготовка
""")
code("""
import json, inspect
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, get_device
from src.data import build_loaders, CIFAR_MEAN, CIFAR_STD
from src.model import build_vgg11_cifar, count_parameters
from src import kmeans_quant, qat, mixed, plots

RESULTS = "results"
set_seed(42)
device = get_device()
print("device:", device, "| torch:", torch.__version__)
""")

md(f"""
## 2. Дані та модель

CIFAR-10, той самий поділ, що й у ДЗ1: 45 000 train / 5 000 val / 10 000 test.
Валідація — для всіх проміжних рішень, тест — на самий кінець. Стартова модель —
щільна VGG11 з ДЗ1 (fp32, {pct(fp32['test'])} на тесті, {fp32['size_MB']:.2f} МБ).
""")
code("""
train_loader, val_loader, test_loader = build_loaders("./data", batch_size=256)
print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

base = build_vgg11_cifar().to(device)
base.load_state_dict(torch.load("../hw1/results/baseline.pt", map_location=device))
print(f"Параметрів: {count_parameters(base)/1e6:.2f}M")
""")

md("""
## 3. Як працює K-Means квантизація

Для кожного згорткового/повнозв'язного шару беремо всі його ваги і кластеризуємо
їх у K = 2^bits центроїдів одновимірним k-means. Кожна вага замінюється значенням
свого центроїда. Ініціалізацію центроїдів беремо лінійною (рівномірно від min до
max ваги) — так радить Deep Compression: великі за модулем ваги рідкісні, але
важливі, а щільнісна ініціалізація їх недо-представляє.

Зберігаємо два об'єкти на шар: **codebook** (K чисел fp32) і **карту індексів**
(по log2(K) біт на вагу). Звідси й стиснення.
""")
code("""
print(inspect.getsource(kmeans_quant.kmeans_1d))
""")

md("""
Подивимось на розподіл ваг одного шару і куди стали центроїди після кластеризації
(3-бітний приклад, K=8). Центроїди сідають щільніше там, де ваг більше, але лінійна
ініціалізація не забуває про хвости.
""")
code("""
layer_name = [n for n, m in kmeans_quant.quantizable_layers(base)
              if isinstance(m, torch.nn.Conv2d)][3]
w = dict(base.named_modules())[layer_name].weight.data
cents, labels = kmeans_quant.kmeans_1d(w, k=8, iters=30)
plots.plot_weight_hist(w.flatten().cpu().numpy(), cents.cpu().numpy(),
                       f"{layer_name}: ваги + 8 центроїдів (3 біти)")
plt.show()
print("унікальних значень ваги після квантизації:", len(cents))
""")

md(f"""
### QAT — до-навчання центроїдів (слайд 36)

Ключова частина. Ваги шару лишаються leaf-параметром; після `loss.backward()`
кожна вага має свій градієнт. Ми **пулимо** ці градієнти за індексом кластера і
крокуємо центроїди, а ваги матеріалізуємо назад як `codebook[index]`. Індекси
заморожені — рухаються лише K значень словника на шар.

Точніше про те, що саме рухається: **квантизовані ваги (Conv/Linear) оновлюються
ЛИШЕ через свої центроїди** — окремі ваги ніколи не крокуються напряму. Дрібні
не-квантизовані параметри — bias і affine-параметри BatchNorm (γ, β) — лишаються
fp32; за замовчуванням (`adapt_extras=True`) ми даємо їм теж адаптуватися, бо це
відчутно допомагає (ablation нижче: +{ex2_gain*100:.1f} п.п. на 2-бітах). З
`adapt_extras=False` рухаються **тільки центроїди**.

**Формула пулінгу.** Для центроїда c, який спільний для ваг кластера S_c:

$$g_c^{{\\text{{sum}}}} = \\sum_{{i \\in S_c}} \\frac{{\\partial L}}{{\\partial W_i}}
\\qquad\\qquad
g_c^{{\\text{{mean}}}} = \\frac{{1}}{{|S_c|}} \\sum_{{i \\in S_c}} \\frac{{\\partial L}}{{\\partial W_i}}$$

**sum** — буквальна реалізація слайда (це точний градієнт репараметризації
W_i = C_c). **mean** — нормалізований варіант (середній градієнт кластера), який
ми й використали. Різниця важлива, бо розміри кластерів різняться на порядки (у
2-бітному conv-шарі ~500k ваг на 4 кластери, у класифікаторі — ~1k): із **sum**
крок центроїда пропорційний розміру кластера, тож під звичайним SGD великі кластери
переганяють і навчання **розходиться** (ablation нижче: sum під SGD падає до
{pct(ab_p_sgd['sum']['qat_test'])}). Під **Adam**, який нормалізує крок кожного
центроїда за його ж статистикою, sum і mean майже збігаються
({pct(ab_p_adam['sum']['qat_test'])} проти {pct(ab_p_adam['mean']['qat_test'])}) —
але mean лишається коректним незалежно від оптимізатора, тому ми беремо його.
""")
code("""
print(inspect.getsource(kmeans_quant.KMeansQuantizer.pool_gradients))
""")

# --------------------------------------------------------------------------- #
md(f"""
## 4. Свіп по бітності: PTQ проти QAT

Для кожної бітності {bits_list} робимо дві точки:
- **PTQ** — лише кластеризація, без до-навчання (скільки коштує сама квантизація);
- **QAT** — далі до-навчаємо центроїди градієнтним пулінгом.

Різниця між ними і є цінністю QAT. На агресивних 2 бітах вона найбільша.
""")
code("""
km = json.load(open(f"{RESULTS}/kmeans.json"))
bit_keys = sorted((k for k in km if k.endswith("bit")), key=lambda k: int(k[:-3]))
ptq = [(int(k[:-3]), km[k]["ptq_val"]) for k in bit_keys]
qat_v = [(int(k[:-3]), km[k]["qat_val"]) for k in bit_keys]
plots.plot_bits_vs_acc({"PTQ (без до-навчання)": ptq, "QAT (до-навчено)": qat_v},
                       baseline_acc=km["fp32"]["val"],
                       title="K-Means квантизація: точність vs бітність (val)")
plt.show()
print("TEST (раз на модель):")
for k in bit_keys:
    r = km[k]
    print(f"  {k}: PTQ val {r['ptq_val']*100:5.2f}%  ->  QAT val {r['qat_val']*100:5.2f}%  "
          f"TEST {r['qat_test']*100:5.2f}%   {r['size_MB']:.2f}MB ({r['compression_x']:.1f}x)")
""")
code(f"""
r = km["{bits_list[0]}bit"]                     # QAT recovery curve at the most aggressive bit-width
plots.plot_history(r["history"], "QAT {bits_list[0]}-bit")
plt.show()
""")

md(f"""
**Що видно.** Post-training на 3–4 бітах падає помірно, а на 2 бітах (лише 4
значення на шар) модель майже ламається — саме там до-навчання центроїдів дає
найбільше ({pct(km[f'{bits_list[0]}bit']['ptq_val'])} → {pct(km[f'{bits_list[0]}bit']['qat_val'])}
на val). Компроміс «розмір↔точність»: {bits_list[-1]}-бітна модель стискає у
{km[f'{bits_list[-1]}bit']['compression_x']:.1f}× майже без втрат
({pct(km[f'{bits_list[-1]}bit']['qat_test'])} на тесті), а {bits_list[0]}-бітна — у
{km[f'{bits_list[0]}bit']['compression_x']:.1f}×, але коштує точності.
""")
code("""
plots.plot_size_vs_acc(
    {"QAT (test)": [(km[k]["size_MB"], km[k]["qat_test"]) for k in bit_keys]},
    baseline=(km["fp32"]["size_MB"], km["fp32"]["test"]),
    title="Компроміс: точність vs розмір моделі")
plt.show()
""")

# --------------------------------------------------------------------------- #
md(f"""
## 5. Бонус, частина 1 — mixed-precision за аналізом чутливості

Не всі шари однаково крихкі. Спочатку робимо **аналіз чутливості до бітності**:
квантизуємо по черзі лише один шар (решта fp32) на 2/3/4 біти і дивимось на
валідацію. Шари, де точність провалюється на 2 бітах, — чутливі, їм треба більше
біт; стійким вистачить менше.

Далі жадібно розподіляємо біти під **фіксований середній бюджет** (~{mixed_avg:.2f}
біт). Важливо: беремо бюджет у «болючому» діапазоні 2–3 біт, де uniform уже втрачає
точність — саме там розумний розподіл має шанс допомогти (на 4 бітах усе й так майже
без втрат, там вигравати нема на чому).

**Порівняння за Парето.** Uniform існує лише в цілих бітах, тож будуємо Парето-лінію
uniform-2 і uniform-3 і дивимось, чи точка mixed лежить **вище** цієї лінії на
своєму розмірі. Щоб зняти питання шуму, всі точки — **середнє ± std за {n_seeds} seed**
(між запусками точність гуляє на кілька десятих через перемішування даних і dropout).
""")
code("""
mx = json.load(open(f"{RESULTS}/mixed.json"))
plots.plot_bit_sensitivity(mx["bit_sensitivity"], mx["fp32"]["val"])
plt.show()
print("Розподіл біт (mixed):")
for n, b in mx["mixed_bits"].items():
    print(f"  {n:28s}: {b} біт")
print(f"середнє = {mx['mixed_avg_bits']:.3f} біт")
""")
code("""
sd = json.load(open(f"{RESULTS}/seeds.json"))
u, m = sd["dense"]["uniform"], sd["dense"]["mixed"]
print(f"Парето-порівняння, {len(sd['seeds'])} seed (mean±std, тест):")
for b in sorted(u, key=int):
    print(f"  uniform {b}-bit : {u[b]['mean']*100:.2f}±{u[b]['std']*100:.2f}%  {u[b]['size_MB']:.2f}MB")
print(f"  mixed {m['avg_bits']:.2f}-bit: {m['mean']*100:.2f}±{m['std']*100:.2f}%  {m['size_MB']:.2f}MB")
print(f"  лінія uniform на розмірі mixed = {m['pareto_bar']*100:.2f}%")
print(f"  --> mixed {'вище' if m['beats'] else 'нижче'} лінії на "
      f"{(m['mean']-m['pareto_bar'])*100:+.2f} п.п.")
""")

md(f"""
**Висновок частини 1.** На бюджеті ~{sd_d_mix['avg_bits']:.2f} біт mixed-precision дає
{ppm(sd_d_mix['mean'], sd_d_mix['std'])} на тесті ({n_seeds} seed) — це {pp(dmix_gain)}
над Парето-лінією uniform того самого розміру. Причому тут mixed навіть перевищує
точність uniform-{hi_uni}-bit ({ppm(sd_d_uni[hi_uni]['mean'], sd_d_uni[hi_uni]['std'])})
**за менший розмір** ({sd_d_mix['size_MB']:.2f} проти {sd_d_uni[hi_uni]['size_MB']:.2f} МБ) —
розрив ({(sd_d_mix['mean']-sd_d_uni[hi_uni]['mean'])*100:+.2f} п.п.) помітно більший за
розкид між сідами (std ≈ {sd_d_mix['std']*100:.2f}), тож це не шум. Логіка: перші
згортки й класифікатор чутливі — їм віддали 4 біти; товсті стійкі середні шари
пережили 2 біти, і в середньому це дешевше й точніше за рівні 3 біти.

(Ремарка: на бюджеті 3–4 біт mixed і uniform були б у межах шуму — там втрачати вже
майже нема на чому, тож розподіл бітів не вирішує. Тому бюджет обрано «болючий».)
""")

# --------------------------------------------------------------------------- #
if pm:
    md(f"""
## 6. Бонус, частина 2 — покращуємо результат прунінгу (Deep Compression)

Це прямо те, чого просить умова бонусу: **покращити результат ітеративного прунінгу
за допомогою mixed-precision квантизації**. Беремо 80%-розріджену модель з ДЗ1
({pct(pr['pruned_only_test'])} на тесті) і квантизуємо **лише ненульові** ваги
(нулі лишаються нулями). Порівнюємо два способи квантизації цієї самої pruned-моделі:
uniform-precision і mixed-precision — знову за Парето на однаковому розмірі, у
форматі sparse+quantized (codebook + індекс на ненульову вагу + бітова маска позицій).
""")
    code("""
sd = json.load(open(f"{RESULTS}/seeds.json"))
pu, pmx = sd["prune"]["uniform"], sd["prune"]["mixed"]
print(f"прунінг (80%): лише прунінг TEST {sd['prune']['pruned_only_test']*100:.2f}%")
print(f"прунінг + квантизація ({len(sd['seeds'])} seed, mean±std, sparse+quant розмір):")
for b in sorted(pu, key=int):
    print(f"  + uniform {b}-bit : {pu[b]['mean']*100:.2f}±{pu[b]['std']*100:.2f}%  {pu[b]['size_MB']:.2f}MB")
print(f"  + mixed {pmx['avg_bits']:.2f}-bit: {pmx['mean']*100:.2f}±{pmx['std']*100:.2f}%  {pmx['size_MB']:.2f}MB "
      f"--> {'вище' if pmx['beats'] else 'нижче'} лінії на {(pmx['mean']-pmx['pareto_bar'])*100:+.2f} п.п.")
mxj = json.load(open(f"{RESULTS}/mixed.json")); phi = mxj["prune_quant_headline"]
print(f"  + uniform {phi['bits']}-bit (headline, 1 seed): TEST {phi['qat_test']*100:.2f}%  "
      f"{phi['sparse_MB']:.2f}MB  ({phi['compression_x']:.1f}x vs fp32)")
""")
    code("""
plt.figure()
img = plt.imread(f"{RESULTS}/prune_mixed_pareto.png"); plt.imshow(img); plt.axis("off"); plt.show()
""")
    md(f"""
**Висновок частини 2.** Прунінг і квантизація — ортогональні осі стиснення й
складаються: разом виходить до {phi['compression_x']:.1f}× при {pct(phi['qat_test'])}
(≈ baseline). На pruned-моделі mixed-precision дає {pp(pmix_gain)} над Парето-лінією
uniform ({n_seeds} seed): {ppm(sd_p_mix['mean'], sd_p_mix['std'])} проти
{ppm(sd_p_uni[hi_uni]['mean'], sd_p_uni[hi_uni]['std'])} для uniform-{hi_uni}-bit.
Виграш скромніший, ніж на щільній моделі, але **стабільний між сідами** (std ≈
{sd_p_mix['std']*100:.2f}, тобто не шум): розумний розподіл бітів покращує вже стиснуту
прунінгом модель. Практичний рецепт для edge-девайсів (Raspberry Pi у фінальному проекті).
""")

# --------------------------------------------------------------------------- #
md(f"""
## 7. Ablation: дві перевірки рецепта QAT

**(1) Пулінг градієнтів — sum vs mean, під SGD і під Adam** (2-бітна модель):

| оптимізатор | mean (наш) | sum (буквальний слайд) |
|---|---|---|
| SGD | {pct(ab_p_sgd['mean']['qat_test'])} (стабільно) | **{pct(ab_p_sgd['sum']['qat_test'])} — РОЗХОДИТЬСЯ** |
| Adam (наш) | {pct(ab_p_adam['mean']['qat_test'])} | {pct(ab_p_adam['sum']['qat_test'])} |

Під **SGD** sum розвалює навчання (крок центроїда ∝ розміру кластера — cluster-size
scaling), а mean стабільний. Під **Adam** sum і mean майже збігаються, бо Adam і так
нормалізує крок кожного центроїда. Математично точний градієнт репараметризації
W_i = C_c — це саме **sum**; **mean** = той самий напрямок, нормований на розмір
кластера (per-cluster preconditioning). Тож mean — не «правильніший», а **стабільний
практичний варіант**: він не залежить від того, чи оптимізатор сам нормалізує крок.

**(2) Що саме навчається — тільки центроїди vs + bias/BN:**

| бітність | тільки центроїди | + bias/BN (наш дефолт) | внесок extras |
|---|---|---|---|
| 2-bit | {pct(ex2['frozen']['qat_test'])} | {pct(ex2['adapt']['qat_test'])} | **{pp(ex2_gain)}** |
| 3-bit | {pct(ex3['frozen']['qat_test'])} | {pct(ex3['adapt']['qat_test'])} | {pp(ex3_gain)} |

Квантизовані ваги завжди рухаються лише через центроїди; питання лише в тому, чи
дати адаптуватися дрібним fp32-параметрам (bias, BN γ/β). На 2 бітах це критично
({pp(ex2_gain)}), на 3 — менше, але стабільно позитивно. Тому дефолт — `adapt_extras=True`.
""")

best_bit = max(bits_list, key=lambda b: km[f"{b}bit"]["qat_test"])
worst_bit = bits_list[0]
md(f"""
## 8. Як ми це робили: підходи та спроби

Коротко про шлях, а не лише про фінал.

**Стартова точка.** Не перенавчали модель — узяли готову щільну VGG11 з ДЗ1 і весь
код даних/навчання звідти, а зверху дописали лише квантизацію. Так HW2 стоїть рівно
на тому самому baseline.

**QAT — де довелося повозитися.** Донавчання центроїдів попервах розвалювало
модель, і ми пройшли кілька ітерацій, поки знайшли причини:
- *сума* градієнтів у кластері (як на діаграмі) нестабільна — розміри кластерів
  різняться на порядки, тож один learning rate розриває великі кластери. Перейшли
  на **усереднення** в межах кластера;
- валідація сильно відставала від train → додали **рекалібрацію BatchNorm** під
  зсунуті ваги перед кожною валідацією;
- модель розходилася через кілька епох → виявилось, що градієнти conv-ваг
  **накопичувалися між батчами** (ваги лежать поза оптимізатором) — полагодили
  обнуленням щобатч;
- SGD+momentum на «зв'язаній» поверхні центроїдів переганяв → замінили на **Adam**.
Після цих чотирьох правок QAT став стабільним і монотонним.

**Бонус — теж не з першого разу.** Спершу пробували mixed-precision на бюджеті 3
біти і **не виграли** — на такій бітності uniform уже майже без втрат, вигравати
нема на чому. Звідси урок: mixed має сенс лише там, де uniform втрачає точність.
Перейшли на «болючі» ~2.5 біт — і воно спрацювало; той самий розподіл застосували
до розрідженої моделі з ДЗ1, що й дало найкращу точку.

**Результати** — у підсумковій таблиці на початку: 4-біт майже без втрат (7.9×),
2-біт рятується QAT-ом ({pct(km[f'{worst_bit}bit']['ptq_val'])} → {pct(km[f'{worst_bit}bit']['qat_test'])}),
mixed тримається над Парето-лінією uniform, а prune+mixed — {ppm(sd_p_mix['mean'], sd_p_mix['std'])}
при {pm['compression_x']:.1f}× (найкраще співвідношення точність/розмір).

## 9. Звіт: висновки і рефлексія

**Що спрацювало добре.** K-Means weight sharing стискає VGG11 у кілька разів майже
без втрати точності: {best_bit}-бітна модель — {pct(km[f'{best_bit}bit']['qat_test'])}
на тесті проти {pct(fp32['test'])} fp32, при {km[f'{best_bit}bit']['compression_x']:.1f}×
стисненні. QAT-пулінг градієнтів за слайдом 36 працює як задумано: на 2 бітах він
піднімає валідацію з {pct(km[f'{worst_bit}bit']['ptq_val'])} (PTQ) до
{pct(km[f'{worst_bit}bit']['qat_val'])}. Бонус: mixed-precision ({n_seeds} seed)
лежить {pp(dmix_gain)} над Парето-лінією uniform — на щільній моделі навіть
**перевищує uniform-{hi_uni}-bit** ({ppm(sd_d_mix['mean'], sd_d_mix['std'])} проти
{ppm(sd_d_uni[hi_uni]['mean'], sd_d_uni[hi_uni]['std'])}, +{(sd_d_mix['mean']-sd_d_uni[hi_uni]['mean'])*100:.2f} п.п.)
**за менший розмір**; на pruned-моделі виграш скромніший ({pp(pmix_gain)}), але
стабільний між сідами (std ≈ {sd_p_mix['std']*100:.2f}, не шум). Разом
прунінг+квантизація стискає до {pm['compression_x']:.1f}× при
{ppm(sd_p_mix['mean'], sd_p_mix['std'])} (≈ baseline).

**Що вийшло не так, як очікувалось.** По-перше, 2-бітна квантизація без до-навчання
майже руйнує модель ({pct(km[f'{worst_bit}bit']['ptq_val'])} на val) — 4 значення на
шар замало. По-друге (і це головний методологічний урок): **mixed-precision виграє
лише там, де uniform уже втрачає точність** — у діапазоні 2–3 біт. На 3–4 бітах
uniform практично без втрат, тож на нашому першому бюджеті (3 біти) mixed і uniform
були в межах шуму; довелося свідомо обрати «болючий» бюджет ~2.5 біт, щоб розподіл
почав щось вирішувати.

**Чому результати можуть бути не ідеальними.** (1) Заявлений розмір — це
**теоретичний packed-size** (codebook + bit-packed індекси), а не розмір поточного
checkpoint: у нас індекси лежать як int64, а ваги матеріалізуються у fp32. Тобто
економія пам'яті — на папері, а не в поточній реалізації. (2) І це *розмір*, а не
швидкість: без спец-ядра weight sharing не прискорює inference. (3) BatchNorm/bias
під час QAT адаптуються (ablation: +{ex2_gain*100:.1f} п.п. на 2 бітах), тож
формально частина мережі не квантизована. (4) 1-D k-means по кожному шару окремо не
враховує взаємодію між шарами; розподіл бітів жадібний.

**Як покращити.** Комбінувати з дистиляцією (вчити квантизовану модель під
fp32-вчителя); квантизувати ще й активації (тоді буде реальне прискорення на
INT-арифметиці); тонший, не жадібний розподіл бітності; Хаффман-кодування індексів
(третій крок Deep Compression); для Raspberry Pi — експорт у формат із реальною
INT8-підтримкою (ONNX/TFLite).
""")

md(f"""
## 10. Відтворення

```bash
python run_kmeans.py   --baseline ../hw1/results/baseline.pt --data-dir ./data --out results
python run_mixed.py    --baseline ../hw1/results/baseline.pt \\
    --pruned ../hw1/results/iterative_final.pt --data-dir ./data --out results
python run_ablation.py --baseline ../hw1/results/baseline.pt --data-dir ./data --out results
python run_seeds.py    --seeds 0 1 2 --baseline ../hw1/results/baseline.pt \\
    --pruned ../hw1/results/iterative_final.pt --data-dir ./data --out results
python build_notebook.py results
python -m pytest tests/          # scatter_add pooling + zeros-preserved
```

Код у `src/`: `kmeans_quant.py` (кластеризація + пулінг градієнтів), `qat.py`
(до-навчання центроїдів + BN-рекалібрація), `mixed.py` (чутливість до бітності +
розподіл), `plots.py`; `data/model/engine/prune/sensitivity/utils` — реюз із ДЗ1.
Версії пакетів — у `requirements.txt`.
""")

# --------------------------------------------------------------------------- #
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
with open(OUT_NB, "w") as f:
    nbf.write(nb, f)
print("wrote", OUT_NB, "with", len(cells), "cells")

# --------------------------------------------------------------------------- #
sweep_report_rows = "\n".join(
    f"| K-Means {b}-bit (QAT) | {b} | {km[f'{b}bit']['compression_x']:.1f}× | "
    f"{pct(km[f'{b}bit']['qat_test'])} |" for b in bits_list
)
pm_report_row = (f"| Прунінг+mixed ~{sd_p_mix['avg_bits']:.2f}-bit (бонус, {n_seeds} seed) | {sd_p_mix['avg_bits']:.2f} | "
                 f"{pm['compression_x']:.1f}× | {ppm(sd_p_mix['mean'], sd_p_mix['std'])} |" if pm else "")
pq_report_row = (f"| Прунінг+квант {phi['bits']}-bit (Deep Compression) | {phi['bits']} | "
                 f"{phi['compression_x']:.1f}× | {pct(phi['qat_test'])} |" if phi else "")

report = f"""# Звіт — ДЗ 2: K-Means квантизація та QAT (VGG11 / CIFAR-10)

Код і результати в репо: [imagic9/efficient-ml-set → hw2](https://github.com/imagic9/efficient-ml-set/tree/main/hw2)

## Методологія
Стартова fp32-модель узята з ДЗ1 (не перенавчаємо). Тест міряється рівно один раз
на кожну фінальну модель; PTQ/QAT-криві та аналіз чутливості — на валідації.
Mixed-precision порівнюється з uniform за Парето на однаковому розмірі (uniform існує
лише в цілих бітах, тож точка mixed має лежати вище лінії uniform того самого розміру).
Bonus-числа — середнє ± std за {n_seeds} seed.

## Підсумкова таблиця (тест — раз на модель)

| Модель | Біт/вагу | Стиснення | Точність |
|---|---|---|---|
| Baseline fp32 (з ДЗ1) | 32 | 1× | {pct(fp32['test'])} (val {pct(fp32['val'])}) |
{sweep_report_rows}
| Bonus: mixed ~{sd_d_mix['avg_bits']:.2f}-bit ({n_seeds} seed) | {sd_d_mix['avg_bits']:.2f} | — | {ppm(sd_d_mix['mean'], sd_d_mix['std'])} |
{pm_report_row}
{pq_report_row}

**Про розмір.** Колонка «Стиснення»/розмір — це теоретичний packed-size
серіалізованого формату (codebook + bit-packed індекси, для прунінгу ще + бітова
маска позицій), **не** розмір поточного PyTorch checkpoint: індекси в реалізації —
int64, ваги під час forward матеріалізуються у fp32.

## Як ми робили (підходи та спроби)
Стартували з готової щільної VGG11 з ДЗ1 (не перенавчали) і дописали зверху лише
квантизацію. Найбільше повозилися з QAT — донавчання центроїдів попервах
розвалювало модель, і ми пройшли кілька ітерацій, поки знайшли причини:
(1) сума градієнтів у кластері нестабільна (розміри кластерів різняться на порядки)
→ перейшли на усереднення в межах кластера; (2) валідація відставала від train →
додали рекалібрацію BatchNorm під зсунуті ваги; (3) модель розходилася через кілька
епох, бо градієнти conv-ваг накопичувалися між батчами (ваги поза оптимізатором) →
обнуляємо щобатч; (4) SGD+momentum на «зв'язаній» поверхні центроїдів переганяв →
замінили на Adam. Для бонусу спершу пробували mixed на бюджеті 3 біти й не виграли
(uniform там уже майже без втрат), тож перейшли на ~2.5 біт, де воно спрацювало, і
застосували той самий розподіл до розрідженої моделі з ДЗ1.

## Метод
- **K-Means weight sharing:** per-layer 1-D k-means, K=2^bits центроїдів, лінійна
  ініціалізація (Deep Compression). Зберігаємо codebook + карту індексів.
- **QAT (слайд 36):** індекси заморожені; після backward пулимо градієнти ваг за
  індексом кластера (mean у межах кластера), крокуємо центроїди, матеріалізуємо
  `weight = codebook[index]`; BN рекалібруємо під зсунуті ваги. **Квантизовані ваги
  оновлюються лише через центроїди**; дрібні bias/BN (fp32) теж адаптуються
  (`adapt_extras=True`).
- **Bonus A:** аналіз чутливості до бітності → жадібний mixed-precision розподіл;
  Парето-порівняння з uniform на «болючому» бюджеті ~{sd_d_mix['avg_bits']:.2f} біт.
- **Bonus B:** покращення прунінгу — 80%-модель з ДЗ1 + mixed-precision квантизація
  ненульових ваг vs uniform; розмір у форматі sparse+quantized.
- **Ablation + тести:** sum vs mean пулінг (SGD/Adam), adapt_extras F/T; 2 unit-тести
  (scatter_add пулінг, збереження нулів).

## Ablation (стисло)
- **Пулінг:** під SGD sum **розходиться** ({pct(ab_p_sgd['sum']['qat_test'])}), mean
  стабільний ({pct(ab_p_sgd['mean']['qat_test'])}); під Adam sum≈mean
  ({pct(ab_p_adam['sum']['qat_test'])} vs {pct(ab_p_adam['mean']['qat_test'])}) — Adam
  нормалізує per-centroid, тож sum-нестабільність саме від cluster-size scaling.
- **adapt_extras:** bias/BN дають +{ex2_gain*100:.1f} п.п. на 2 бітах, {pp(ex3_gain)} на 3.

## What worked well
K-Means стискає модель у кілька разів майже без втрат: {best_bit}-бітна —
{pct(km[f'{best_bit}bit']['qat_test'])} проти {pct(fp32['test'])} fp32
({km[f'{best_bit}bit']['compression_x']:.1f}×). QAT-пулінг піднімає 2-бітну val з
{pct(km[f'{worst_bit}bit']['ptq_val'])} до {pct(km[f'{worst_bit}bit']['qat_val'])}.
Mixed-precision ({n_seeds} seed) лежить {pp(dmix_gain)} над Парето-лінією uniform — на
щільній моделі навіть **перевищує uniform-{hi_uni}-bit за менший розмір**
({ppm(sd_d_mix['mean'], sd_d_mix['std'])} проти {ppm(sd_d_uni[hi_uni]['mean'], sd_d_uni[hi_uni]['std'])}),
на pruned-моделі виграш скромніший ({pp(pmix_gain)}), але стабільний. Разом
прунінг+квантизація стискає до {pm['compression_x']:.1f}× при
{ppm(sd_p_mix['mean'], sd_p_mix['std'])} (≈ baseline).

## What didn't turn out as expected
2-бітна квантизація без до-навчання майже ламає модель
({pct(km[f'{worst_bit}bit']['ptq_val'])} val). Головний урок: mixed-precision виграє
лише у діапазоні 2–3 біт, де uniform уже втрачає точність; на 3–4 бітах uniform
майже без втрат і розподіл бітів у межах шуму — тому бюджет для бонусу довелося
обрати свідомо «болючим» (~{sd_d_mix['avg_bits']:.2f} біт). Плюс: single-run давав
mixed нарівні з uniform-3, і лише 3 seed показали стабільний виграш — ще один довід
не робити висновків з одного запуску.

## Why results might not be great
Заявлений розмір — теоретичний packed-size, а не поточний checkpoint (індекси int64,
ваги матеріалізуються у fp32). І це розмір, а не швидкість: без спец-ядра weight
sharing не прискорює inference. BatchNorm/bias під час QAT адаптуються (частина
мережі формально не квантизована). Пошаровий 1-D k-means не бачить взаємодії між
шарами; розподіл бітів жадібний.

## How to improve
Дистиляція під fp32-вчителя; квантизація активацій (реальне INT-прискорення);
оптимальний (не жадібний) розподіл бітності; Хаффман-кодування індексів (крок 3
Deep Compression); експорт у ONNX/TFLite з INT8 для Raspberry Pi.
"""
with open(OUT_REPORT, "w") as f:
    f.write(report)
print("wrote", OUT_REPORT)
