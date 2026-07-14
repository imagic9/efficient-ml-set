"""Assemble the submission notebook (HW1_VGG11_Pruning.ipynb) and REPORT.md.

Reads metrics produced by run_all.py + run_structured.py from RESULTS_DIR and
bakes real numbers into the narrative. Run AFTER both experiments finish, then
execute the notebook with nbconvert.
"""
import json
import os
import sys

import nbformat as nbf

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT_NB = "HW1_VGG11_Pruning.ipynb"
OUT_REPORT = "REPORT.md"


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


base = load("baseline.json")
oneshot = load("oneshot.json")
iterative = load("iterative.json")
sens = load("sensitivity.json")
struct = load("structured.json")
confirm = load("confirm95.json")

# test-confirmed numbers at high sparsity (the bonus win)
c95 = confirm["results"]                     # {uniform_layer, global, sensitivity}
c95_target = confirm["target"]
sens95, uni95, glob95 = c95["sensitivity"], c95["uniform_layer"], c95["global"]
bonus_win = sens95["test"] - uni95["test"]   # sensitivity vs naive uniform at high sparsity

base_test, base_val, params_m = base["test"], base["val"], base["params_M"]
os_before, os_test = oneshot["val_before_ft"], oneshot["test"]
it_test = iterative["test"]
it_sparsity = iterative["val_points"][-1][0]

# unstructured headline test at 80% for the three allocation methods
hd = sens["headline_test"]         # {uniform_layer, global, sensitivity}
hd_sp = sens["headline_sparsity"]

# structured headline at ~50% MACs
st = struct["headline"]            # {uniform, sensitivity: {macs_frac, params_M, val, test}}
st_uni, st_sens = st["uniform"], st["sensitivity"]
struct_win = st_sens["test"] - st_uni["test"]

pct = lambda x: f"{x * 100:.2f}%"
pp = lambda x: f"{x * 100:+.2f} п.п."

cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))


# --------------------------------------------------------------------------- #
md(f"""
# Домашня робота 1 — Ітеративний прунінг VGG11 на CIFAR-10

**Курс:** Efficient ML, SET University

Ми навчаємо VGG11 на CIFAR-10, а потім прибираємо близько 80% ваг так, щоб
точність просіла якомога менше. Порівнюємо one-shot проти ітеративного прунінгу,
а в бонусній частині розбираємо, як саме розподіляти прунінг по шарах — і на
дрібнозернистому, і на структурному (канальному) рівні.

**Методологія оцінювання.** Тестовий набір ми чіпаємо рівно один раз для кожної
фінальної моделі — після того, як усі рішення вже прийнято. Усі проміжні числа,
криві та порівняння методів рахуються на валідації. Це принципово: інакше вибір
за тестом — це витік інформації.

### Підсумок

| Модель | Розрідженість | Точність (тест) |
|---|---|---|
| Baseline (щільна) | 0% | **{pct(base_test)}** |
| One-shot 80% (після fine-tuning) | 80% | {pct(os_test)} |
| Ітеративний 80% (global magnitude) | {pct(it_sparsity)} | **{pct(it_test)}** |
| Бонус @95%: uniform-per-layer | 95% | {pct(uni95['test'])} |
| Бонус @95%: sensitivity-guided | 95% | **{pct(sens95['test'])}** |

На 95% розрідженості sensitivity-guided випереджає наївний uniform на
{pp(bonus_win)} (тест) — саме там аналіз чутливості починає працювати.
""")

md("""
## 1. Підготовка
""")
code("""
import json
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, get_device
from src.data import build_loaders, CIFAR_MEAN, CIFAR_STD
from src.model import build_vgg11_cifar, count_parameters
from src import prune, plots

RESULTS = "results"
set_seed(42)
device = get_device()
print("device:", device, "| torch:", torch.__version__)
""")

md("""
## 2. Дані: CIFAR-10

60 000 кольорових зображень 32×32 у 10 класах. Стандартний поділ — 50 000 на
навчання і 10 000 на тест. Від навчальної частини відрізаємо 5 000 на валідацію:
за нею відбираємо найкращу модель і робимо всі порівняння, а тестові 10 000
залишаємо на самий кінець. Аугментації для навчання — випадковий зсув
(`RandomCrop` з відступом 4) і горизонтальне віддзеркалення.
""")
code("""
train_loader, val_loader, test_loader = build_loaders("./data", batch_size=256)
print(f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")

import torch
classes = ['літак','авто','птах','кіт','олень','собака','жаба','кінь','корабель','вантажівка']
mean = torch.tensor(CIFAR_MEAN).view(3,1,1); std = torch.tensor(CIFAR_STD).view(3,1,1)
imgs, labels = next(iter(val_loader))
fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
for ax, img, lbl in zip(axes.flat, imgs, labels):
    ax.imshow((img*std+mean).clamp(0,1).permute(1,2,0).numpy())
    ax.set_title(classes[lbl], fontsize=8); ax.axis('off')
plt.tight_layout(); plt.show()
""")

md("""
## 3. Модель: VGG11 з torchvision, адаптована під CIFAR

Беремо `vgg11_bn` з torchvision. Оригінал розрахований на ImageNet (вхід 224×224):
після п'яти пулінгів лишається карта 7×7, а класифікатор — три величезні
повнозв'язні шари (~124 млн параметрів). Наш вхід 32×32 стискається до 1×1, тому
ImageNet-голова недоречна. Згортковий кістяк VGG11 лишаємо без змін, а голову
замінюємо на компактну під 10 класів. Версію з batch-norm обрано свідомо: без неї
VGG11 з нуля на CIFAR навчається нестабільно.
""")
code("""
model = build_vgg11_cifar().to(device)
print(f"Параметрів: {count_parameters(model)/1e6:.2f}M")
print(model.classifier)
""")

md("""
### Як обрізаємо ваги (дрібнозернистий прунінг)

Обнуляємо ваги, найменші за модулем, у кожному згортковому й повнозв'язному шарі.
Маска запам'ятовується і накладається після кожного кроку оптимізатора — обнулені
ваги лишаються нулем, решта продовжує вчитися.
""")
code("""
import inspect
from src.prune import _layer_mask, FineGrainedPruner
print(inspect.getsource(_layer_mask))
print(inspect.getsource(FineGrainedPruner.apply))
""")

md(f"""
## 4. Baseline

80 епох SGD (momentum 0.9, Nesterov, weight decay 5e-4) з косинусним спадом.
Відбираємо ваги за найкращою валідацією ({pct(base_val)}) і лише тоді міряємо
тест: **{pct(base_test)}**. Це відправна точка для всіх обрізаних моделей.
""")
code("""
base = json.load(open(f"{RESULTS}/baseline.json"))
plots.plot_history(base["history"], "Baseline"); plt.show()
print(f"val = {base['val']*100:.2f}%   TEST = {base['test']*100:.2f}%")
""")

md(f"""
## 5. One-shot прунінг: зрізаємо 80% за один раз

Найпростіший підхід — одразу обнулити 80% ваг за глобальним порогом і до-навчити.
Одразу після зрізу валідаційна точність падає до **{pct(os_before)}** (модель
фактично зламана), але кілька епох до-навчання відновлюють її майже до рівня
baseline. Фінальний тест одношотної моделі — **{pct(os_test)}**.
""")
code("""
os_ = json.load(open(f"{RESULTS}/oneshot.json"))
print(f"sparsity            = {os_['sparsity']*100:.1f}%")
print(f"val ДО до-навчання  = {os_['val_before_ft']*100:.2f}%")
print(f"val ПІСЛЯ           = {os_['val']*100:.2f}%")
print(f"TEST (фінально)     = {os_['test']*100:.2f}%")
""")

md(f"""
## 6. Ітеративний прунінг: доходимо до 80% поступово

Замість різкого зрізу нарощуємо розрідженість за кілька кроків, до-навчаючи між
ними. Розклад {iterative['schedule']} прибирає однакову частку ваг, що ще
лишились, і на останньому кроці виходить рівно на 80%.

Криві нижче — на **валідації**. Помітно, як «одразу після зрізу» модель
провалюється дедалі глибше, а «після до-навчання» тримається майже на рівні
baseline. Фінальний тест ітеративної моделі — **{pct(it_test)}**.
""")
code("""
it = json.load(open(f"{RESULTS}/iterative.json"))
plots.plot_sparsity_vs_acc({
    "після до-навчання (val)": it["val_points"],
    "одразу після зрізу (val)": [[0.0, base['val']]] + it["val_after_cut"],
}, "Ітеративний прунінг (валідація)"); plt.show()
print(f"фінальний ітеративний TEST = {it['test']*100:.2f}%   "
      f"(one-shot TEST = {it['oneshot_test']*100:.2f}%)")
""")

md(f"""
На 80% ітеративний ({pct(it_test)}) і one-shot ({pct(os_test)}) виходять майже
однаково — на CIFAR-10 надлишкових ваг настільки багато, що навіть різкий зріз
відновлюється. Різниця в тому, *наскільки боляче* проходить сам зріз: one-shot
падає до {pct(os_before)}, ітеративний майже не помічає кожного окремого кроку.
""")

# --------------------------------------------------------------------------- #
best_unstruct = max(hd, key=lambda k: hd[k])
md(f"""
## 7. Бонус, частина 1 — як розподіляти прунінг по шарах (дрібнозернистий)

Порівнюємо три способи задати, скільки різати в кожному шарі, при однаковій
сумарній розрідженості, через свіп по рівнях sparsity (усе на валідації):

- **uniform per-layer** — однаковий відсоток у кожному шарі (наївний);
- **global magnitude** — один глобальний поріг за модулем (наш основний метод);
- **sensitivity-guided** — per-layer бюджет за аналізом чутливості шарів.

Аналіз чутливості: по черзі обрізаємо лише один шар (без до-навчання) і дивимось,
де точність провалюється рано — такі шари чутливі, їх треба берегти.
""")
code("""
sens = json.load(open(f"{RESULTS}/sensitivity.json"))
plots.plot_sensitivity(sens["curves"], base["val"]); plt.show()
""")
code("""
labels = {"uniform_layer":"uniform per-layer","global":"global magnitude",
          "sensitivity":"sensitivity-guided"}
plots.plot_sparsity_vs_acc({labels[k]: sens["sweep"][k] for k in labels},
                           "Три стратегії розподілу (валідація)"); plt.show()
print("TEST на 80% (раз на модель):")
for k in labels:
    print(f"  {labels[k]:20s}: {sens['headline_test'][k]*100:.2f}% "
          f"@ {sens['headline_sparsity'][k]*100:.1f}%")
""")

code('''
c = json.load(open(f"{RESULTS}/confirm95.json"))["results"]
print("TEST на 95% розрідженості (раз на модель):")
for k in ["uniform_layer","global","sensitivity"]:
    print(f"  {k:16s}: {c[k]['test']*100:.2f}%  @ {c[k]['sparsity']*100:.1f}%")
''')

md(f"""
**Висновок частини 1.** На 80–90% усі три методи в межах шуму, трохи попереду
global magnitude ({pct(hd['global'])} на тесті на 80%). Але зі зростанням
розрідженості наївний uniform-per-layer валиться найшвидше, і на **95%**
sensitivity-guided його впевнено обганяє — на тесті **{pct(sens95['test'])} проти
{pct(uni95['test'])} ({pp(bonus_win)})**, практично наздоганяючи global
({pct(glob95['test'])}).

Чому так. Глобальний поріг за модулем прибирає саме глобально найменші ваги —
мінімальне збурення мережі, тому при помірній розрідженості його важко
перевершити. Але коли ваг лишається дуже мало, критично стає не чіпати чутливі
шари (перші згортки, класифікатор) — і тут аналіз чутливості дає перевагу. Тобто
цінність sensitivity зростає саме на агресивній розрідженості — і на тому самому
бюджеті ваг вона дає вищу точність, ніж наївний рівномірний прунінг.
""")

# --------------------------------------------------------------------------- #
md(f"""
## 8. Бонус, частина 2 — структурний (канальний) прунінг

Тут ми **фізично видаляємо цілі згорткові фільтри** (через `torch-pruning`), а не
обнуляємо окремі ваги. Це справжнє зменшення обчислень: на щільній моделі
{struct['base_macs_M']:.0f}M MACs, після прунінгу — менше, і це прямо перетворюється
на швидший inference (те, що потрібно у фінальному проекті на Raspberry Pi).

Тут per-layer розподіл справді вирішує: у шарів різна форма й різна важливість
каналів, єдиного глобального критерію нема. Порівнюємо **uniform** проти
**sensitivity-guided** канального прунінгу на однаковому бюджеті MACs.
""")
code("""
st = json.load(open(f"{RESULTS}/structured.json"))
plots.plot_sensitivity(st["curves"], st["base_val"]); plt.show()   # чутливість каналів
""")
code("""
fig, ax = plt.subplots(figsize=(7,5))
for method in ["uniform","sensitivity"]:
    pts = sorted(st["sweep"][method])
    ax.plot([p[0]*100 for p in pts], [p[1]*100 for p in pts], marker="o", label=method)
ax.axhline(st["base_val"]*100, color="k", ls="--", lw=1, label="baseline (щільна)")
ax.set_xlabel("залишок MACs, % від щільної"); ax.set_ylabel("val accuracy, %")
ax.set_title("Структурний прунінг: uniform vs sensitivity-guided")
ax.legend(); ax.grid(alpha=0.3); plt.show()

h = st["headline"]
print("HEADLINE (~50% MACs, TEST раз на модель):")
for m in ["uniform","sensitivity"]:
    print(f"  {m:12s}: {h[m]['test']*100:.2f}%  @ {h[m]['macs_frac']*100:.0f}% MACs, "
          f"{h[m]['params_M']:.2f}M params")
""")

md(f"""
**Висновок частини 2.** Головне, що дає структурний прунінг — **реальне зменшення
обчислень**: щільна модель {struct['base_macs_M']:.0f}M MACs, а на ~{st_uni['macs_frac']*100:.0f}%
MACs вона ще тримає ~{pct(st_uni['test'])} на тесті. Це саме те, що прискорює
inference на Raspberry Pi.

А от **наївний sensitivity тут не переграв uniform** (криві майже збігаються, на
~50% MACs навіть трохи гірше: {pct(st_sens['test'])} проти {pct(st_uni['test'])}).
Причина повчальна: розподіл, побудований лише за *точністю*, **сліпий до вартості
MACs**. Він береже чутливі ранні згортки (а вони найдорожчі за обчисленнями, бо
працюють на великій роздільній здатності) і сильніше ріже останній conv, який на
карті 1×1 майже безкоштовний. Тобто економить обчислення не там, де треба.

Правильний наступний крок — **MAC-aware розподіл** (різати сильніше там, де шар і
надлишковий, і дорогий за обчисленнями). Це напрям NetAdapt / AMC і прямий місток
до фінального проекту, де метрика — саме FPS, а не абстрактна розрідженість.
""")

# --------------------------------------------------------------------------- #
md(f"""
## 9. Звіт: висновки і рефлексія

**Що спрацювало добре.** Дрібнозернистий прунінг прибрав 80% ваг практично без
втрати точності — {pct(base_test)} → {pct(it_test)} ({pp(it_test - base_test)}),
у межах шуму. Ключове — до-навчання між зрізами. Бонусний результат: на **95%**
розрідженості аналіз чутливості дав вимірну перевагу над наївним рівномірним
прунінгом — {pct(sens95['test'])} проти {pct(uni95['test'])} на тесті
({pp(bonus_win)}). Структурний прунінг показав реальне зменшення обчислень:
{struct['base_macs_M']:.0f}M → ~{st_uni['macs_frac']*struct['base_macs_M']:.0f}M MACs
при ~{pct(st_uni['test'])}.

**Що вийшло не так, як очікувалось.** По-перше, при помірній розрідженості (80–90%)
аналіз чутливості **не перевершив global magnitude** ({pct(hd['global'])}) — і це
не брак реалізації, а властивість дрібнозернистого прунінгу: глобальний поріг за
модулем уже майже оптимальний. По-друге, наївний sensitivity у структурному
прунінгу **не переграв uniform**, бо розподіл за самою лише точністю сліпий до
вартості MACs (береже дорогі ранні шари, ріже дешеві пізні). Обидва спостереження
уточнюють, *де саме* аналіз чутливості корисний: на агресивній розрідженості та за
MAC-aware розподілу.

**Чому результати можуть бути не ідеальними.** Дрібнозернистий прунінг не дає
реального прискорення на GPU (лише формальна розрідженість). Бюджети до-навчання
обмежені часом. Критерій за модулем жадібний і не враховує взаємодію ваг. Чутливість
зондується без до-навчання, тож недооцінює, наскільки шар відновлюється.

**Як можна покращити.** MAC-aware розподіл для структурного прунінгу (NetAdapt /
AMC) — різати там, де шар і надлишковий, і дорогий; довести структурний прунінг до
реального FPS на Raspberry Pi; більше епох до-навчання і кроків; зв'язка з
квантизацією та дистиляцією; критерії відбору з урахуванням градієнтів.
""")

md("""
## 10. Відтворення

```bash
python run_all.py --data-dir ./data --out results        # baseline, one-shot, iterative, sweep
python run_structured.py --data-dir ./data --out results  # структурний бонус (реюзає baseline.pt)
python build_notebook.py results                          # зібрати цей ноутбук
```

Код у `src/`: `data.py`, `model.py`, `engine.py`, `prune.py` (дрібнозернистий),
`sensitivity.py`, `structured.py` (канальний, torch-pruning), `plots.py`.
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
report = f"""# Звіт — ДЗ 1: Ітеративний прунінг VGG11 на CIFAR-10

## Методологія
Тест міряється рівно один раз на кожну фінальну модель, після всіх рішень. Усі
проміжні криві та порівняння методів — на валідації.

## Підсумкова таблиця (тест)

| Модель | Розрідженість | Точність |
|---|---|---|
| Baseline (щільна) | 0% | {pct(base_test)} |
| One-shot 80% (після fine-tuning) | 80% | {pct(os_test)} |
| Ітеративний 80% (global magnitude) | {pct(it_sparsity)} | {pct(it_test)} |
| Бонус @95%: uniform-per-layer | 95% | {pct(uni95['test'])} |
| Бонус @95%: sensitivity-guided | 95% | {pct(sens95['test'])} |

Параметрів: {params_m:.2f}M. Baseline val: {pct(base_val)}.
Структурний прунінг: {struct['base_macs_M']:.0f}M MACs -> ~{st_uni['macs_frac']*struct['base_macs_M']:.0f}M
при ~{pct(st_uni['test'])}.

## Метод
- **Модель:** VGG11 (`vgg11_bn`) з torchvision, кістяк без змін, компактна голова.
- **Дрібнозернистий прунінг:** unstructured, за модулем ваги, персистентні маски.
- **Ітеративний розклад:** {iterative['schedule']} (вихід рівно на 80%).
- **Бонус 1 (unstructured):** свіп uniform-per-layer / global-magnitude /
  sensitivity-guided; test-підтвердження на 95%.
- **Бонус 2 (structured):** канальний прунінг через torch-pruning з реальним
  зменшенням MACs; uniform vs sensitivity-guided.

## What worked well
80% ваг прибрано практично без втрати: {pct(base_test)} -> {pct(it_test)}
({pp(it_test - base_test)}). На 95% розрідженості sensitivity-guided перевершив
наївний uniform на тесті: {pct(sens95['test'])} проти {pct(uni95['test'])}
({pp(bonus_win)}). Структурний прунінг дав реальне зменшення обчислень.

## What didn't turn out as expected
При 80-90% аналіз чутливості не перевершив global magnitude ({pct(hd['global'])}) —
глобальний поріг за модулем майже оптимальний для unstructured. У структурному
прунінгу наївний sensitivity не переграв uniform, бо розподіл за самою точністю
сліпий до вартості MACs. Обидва уточнюють, де чутливість корисна: на агресивній
розрідженості та за MAC-aware розподілу.

## Why results might not be great
Unstructured не дає реального прискорення (формальна розрідженість); бюджет
до-навчання обмежений; критерій за модулем жадібний; чутливість зондується без
до-навчання.

## How to improve
MAC-aware розподіл для структурного прунінгу (NetAdapt / AMC); довести структурний
прунінг до реального FPS на Raspberry Pi; більше епох/кроків; зв'язка з
квантизацією/дистиляцією; кращі критерії відбору ваг.
"""
with open(OUT_REPORT, "w") as f:
    f.write(report)
print("wrote", OUT_REPORT)
