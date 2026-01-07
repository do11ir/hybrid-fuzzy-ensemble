# hybrid-fuzzy-ensemble
for running the venv in projext dir use:

```bash
.venv\Scripts\activate
```

libs already installed in requirements.txt after adding new libs pls use :

```bash
pip freeze > requirements.txt
```

file structure :

hybrid-fuzzy-ensemble
│
├── data
│   ├── raw
│   └── processed
│
├── src
│   ├── __init__.py
│   ├── data_loader.py
│   ├── models.py
│   ├── fuzzy.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── results
│   └── metrics.csv   (بعداً ساخته می‌شود)
│
├── run.py
├── requirements.txt
└── README.md


dataset link and info in :
https://archive.ics.uci.edu/dataset/45/heart+disease


هدف

ایجاد یک شبکه عصبی ترکیبی (Hybrid Ensemble) برای پیش‌بینی Heart Disease که شامل:

دو شبکه عصبی پایه (BaseNN1 و BaseNN2)

لایه فازی Voting برای ترکیب خروجی‌ها با وزن‌دهی بر اساس اعتماد (confidence)

شبکه نهایی (Meta-Network) که خروجی فازی را پردازش می‌کند

هدف اصلی: افزایش دقت و پایداری مدل نسبت به استفاده از یک شبکه تنها.

2. ساختار مدل
2.1 BaseNN1

عمق کم: 2 لایه مخفی

Activation: ReLU

Dropout: 0.2

خروجی: Sigmoid (برای احتمال حضور بیماری)

2.2 BaseNN2

عمق بیشتر: 3 لایه مخفی

Activation: Tanh

Dropout: 0.1

خروجی: Sigmoid

2.3 Voting Layer (Fuzzy)

ترکیب خروجی دو شبکه

وزن‌دهی بر اساس confidence: هر خروجی نزدیک به 0 یا 1 وزن بیشتری دارد

استفاده از قواعد ساده فازی:

اگر هر دو شبکه نزدیک به 0 یا 1 باشند، نتیجه نهایی همان باشد

اگر شبکه‌ها اختلاف داشته باشند، میانگین فازی گرفته شود

2.4 Meta Network (Final Layer)

یک لایه Linear ساده برای خروجی نهایی

Activation: Sigmoid

3. مزایا

استفاده از دو شبکه متفاوت باعث می‌شود هر شبکه بخش‌های متفاوتی از ویژگی‌ها را یاد بگیرد

Ensemble کاهش خطای واریانس و overfitting را فراهم می‌کند

Voting Layer فازی باعث تصمیم‌گیری هوشمندتر و مقاوم‌تر می‌شود




این دیتاست از UCI Machine Learning Repository گرفته شده است و هدف آن تشخیص حضور یا عدم حضور بیماری قلبی است.
UCI Machine Learning Repository

2. تعداد رکوردها و ویژگی‌ها

شامل 303 رکورد از بیماران است.
UCI Machine Learning Repository

دارای 76 ویژگی در فایل اصلی، اما مدل‌های پژوهشی معمولاً از 14 ویژگی اصلی استفاده می‌کنند.
UCI Machine Learning Repository

3. متغیر هدف (Target)

مقدار هدف در دیتاست به شکل عدد از 0 تا 4 است که:

0 = عدم بیماری (Healthy)

1–4 = نشانه‌های مختلف بیماری قلبی

معمولاً برای مساله‌ی طبقه‌بندی دودویی (Binary Classification)، مقادیر 1 تا 4 به 1 تبدیل می‌شوند یعنی وجود بیماری؛ و 0 همان عدم وجود بیماری.
archive-beta.ics.uci.edu

4. برخی از ویژگی‌های مهم

دیتاست شامل ویژگی‌های پزشکی کلی مثل سن، فشار خون، کلسترول، نوع درد قفسه سینه و … است.
IBM Cloud Pak for Data

برخی ویژگی‌های معمول (از مجموعه 14 ویژگی انتخاب‌شده):

age: سن فرد

sex: جنسیت (۱=مرد، ۰=زن)

cp: نوع درد قفسه‌ی سینه

trestbps: فشار خون در حالت استراحت

chol: کلسترول سرم

fbs: قند خون ناشتا

restecg: علائم ECG استراحت

thalach: حداکثر ضربان قلب

exang: آنژین القا شده با ورزش

oldpeak: ST depression induced by exercise

slope, ca, thal: ویژگی‌های دیگر مربوط به ECG و مقادیر تست‌های پزشکی

num: متغیر هدف (0 تا 4)
IBM Cloud Pak for Data





معماری پیشنهادی (Proposed Architecture)

مدل پیشنهادی از سه بخش اصلی تشکیل شده است:

3.1 شبکه اول (BaseNN1 – شبکه کم‌عمق)

این شبکه به‌گونه‌ای طراحی شده که الگوهای ساده و خطی‌تر را از داده یاد بگیرد.

ویژگی‌ها:

2 لایه مخفی

تابع فعال‌سازی: ReLU

Dropout = 0.2

خروجی: Sigmoid (احتمال وجود بیماری)

هدف این شبکه:

یادگیری الگوهای پایه و عمومی بدون پیچیدگی زیاد

3.2 شبکه دوم (BaseNN2 – شبکه عمیق‌تر)

این شبکه نسبت به BaseNN1 عمیق‌تر است و با تابع فعال‌سازی متفاوت کار می‌کند تا فضای جستجوی متفاوتی را پوشش دهد.

ویژگی‌ها:

3 لایه مخفی

تابع فعال‌سازی: Tanh

Dropout = 0.1

خروجی: Sigmoid

هدف این شبکه:

یادگیری روابط غیرخطی‌تر و پیچیده‌تر بین ویژگی‌ها

استفاده از Tanh باعث می‌شود رفتار این شبکه از نظر توزیع گرادیان و پاسخ متفاوت از BaseNN1 باشد، که برای ensemble بسیار مهم است.

3.3 لایه رأی‌گیری فازی (Fuzzy Voting Layer)

نوآوری اصلی مدل در این بخش قرار دارد.

ایده اصلی:

به‌جای میانگین‌گیری ساده از خروجی شبکه‌ها، از یک وزن‌دهی مبتنی بر confidence استفاده شده است.

تعریف confidence:

اگر خروجی یک شبکه به 0 یا 1 نزدیک باشد → اعتماد بیشتر
اگر خروجی نزدیک به 0.5 باشد → عدم قطعیت بیشتر

فرمول استفاده‌شده:

confidence = |output - 0.5| × 2

نحوه ترکیب:
Weighted Output = (x1 × conf1 + x2 × conf2) / (conf1 + conf2)


نتیجه:

اگر هر دو شبکه مطمئن باشند → تصمیم قوی

اگر یکی مطمئن‌تر باشد → رأی آن غالب می‌شود

اگر هر دو نامطمئن باشند → تصمیم نرم و محتاطانه

این رفتار دقیقاً مشابه تصمیم‌گیری فازی انسانی است.

3.4 شبکه نهایی (Meta Network)

خروجی لایه فازی وارد یک لایه خطی ساده می‌شود:

Linear(1 → 1)

Sigmoid

هدف:

تنظیم نهایی خروجی ensemble و افزایش انعطاف‌پذیری تصمیم نهایی

4. فرآیند آموزش و ارزیابی

Optimizer: Adam

Loss Function: Binary Cross Entropy

Epochs: 50

Batch Size: 8

مدل روی داده‌های آموزش train شده و روی validation ارزیابی شده است.

5. نتایج تجربی (Experimental Results)

نتایج نشان می‌دهد که مدل:

Accuracy ≈ 85%

Recall برای کلاس بیمار ≈ 97%

خطای False Negative بسیار پایین

Confusion Matrix نمونه:

[[20  8]
 [ 1 32]]


این موضوع در کاربردهای پزشکی اهمیت زیادی دارد، زیرا از دست دادن بیمار (False Negative) هزینه بالایی دارد.

6. تحلیل و جمع‌بندی

مزایای مدل پیشنهادی:

کاهش واریانس نسبت به شبکه تکی

تصمیم‌گیری پایدارتر

استفاده از confidence در فرآیند ensemble

مناسب برای دیتاست‌های پزشکی کوچک

مدل پیشنهادی یک چارچوب منعطف است و می‌تواند به‌راحتی:

به شبکه‌های بیشتر گسترش یابد

با قوانین فازی پیچیده‌تر ترکیب شود

برای مسائل پزشکی دیگر استفاده شود



(.venv) C:\Users\ME\Desktop\hybrid-fuzzy-ensemble>python run-cross-validate-5.py

================ FOLD 1 / 5 ================
Epoch [1/50] | Train Loss: 0.6357 | Val Loss: 0.6064 | Val Acc: 0.4590
Epoch [2/50] | Train Loss: 0.5881 | Val Loss: 0.5383 | Val Acc: 0.6393
Epoch [3/50] | Train Loss: 0.5490 | Val Loss: 0.5159 | Val Acc: 0.8197
Epoch [4/50] | Train Loss: 0.5323 | Val Loss: 0.5057 | Val Acc: 0.8525
Epoch [5/50] | Train Loss: 0.5233 | Val Loss: 0.4999 | Val Acc: 0.8525
Epoch [6/50] | Train Loss: 0.5172 | Val Loss: 0.4925 | Val Acc: 0.8852
Epoch [7/50] | Train Loss: 0.5101 | Val Loss: 0.4857 | Val Acc: 0.9016
Epoch [8/50] | Train Loss: 0.5061 | Val Loss: 0.4795 | Val Acc: 0.9016
Epoch [9/50] | Train Loss: 0.5005 | Val Loss: 0.4771 | Val Acc: 0.9016
Epoch [10/50] | Train Loss: 0.4979 | Val Loss: 0.4706 | Val Acc: 0.9016
Epoch [11/50] | Train Loss: 0.4933 | Val Loss: 0.4678 | Val Acc: 0.9180
Epoch [12/50] | Train Loss: 0.4884 | Val Loss: 0.4629 | Val Acc: 0.9180
Epoch [13/50] | Train Loss: 0.4846 | Val Loss: 0.4593 | Val Acc: 0.9180
Epoch [14/50] | Train Loss: 0.4816 | Val Loss: 0.4559 | Val Acc: 0.9180
Epoch [15/50] | Train Loss: 0.4785 | Val Loss: 0.4519 | Val Acc: 0.9180
Epoch [16/50] | Train Loss: 0.4754 | Val Loss: 0.4471 | Val Acc: 0.9180
Epoch [17/50] | Train Loss: 0.4727 | Val Loss: 0.4433 | Val Acc: 0.9180
Epoch [18/50] | Train Loss: 0.4702 | Val Loss: 0.4420 | Val Acc: 0.9180
Epoch [19/50] | Train Loss: 0.4668 | Val Loss: 0.4364 | Val Acc: 0.9180
Epoch [20/50] | Train Loss: 0.4644 | Val Loss: 0.4330 | Val Acc: 0.9180
Epoch [21/50] | Train Loss: 0.4633 | Val Loss: 0.4302 | Val Acc: 0.9180
Epoch [22/50] | Train Loss: 0.4598 | Val Loss: 0.4288 | Val Acc: 0.9180
Epoch [23/50] | Train Loss: 0.4579 | Val Loss: 0.4256 | Val Acc: 0.9180
Epoch [24/50] | Train Loss: 0.4540 | Val Loss: 0.4242 | Val Acc: 0.9180
Epoch [25/50] | Train Loss: 0.4520 | Val Loss: 0.4242 | Val Acc: 0.9180
Epoch [26/50] | Train Loss: 0.4499 | Val Loss: 0.4210 | Val Acc: 0.9344
Epoch [27/50] | Train Loss: 0.4472 | Val Loss: 0.4176 | Val Acc: 0.9344
Epoch [28/50] | Train Loss: 0.4464 | Val Loss: 0.4162 | Val Acc: 0.9180
Epoch [29/50] | Train Loss: 0.4437 | Val Loss: 0.4126 | Val Acc: 0.9180
Epoch [30/50] | Train Loss: 0.4397 | Val Loss: 0.4109 | Val Acc: 0.9180
Epoch [31/50] | Train Loss: 0.4383 | Val Loss: 0.4099 | Val Acc: 0.9180
Epoch [32/50] | Train Loss: 0.4361 | Val Loss: 0.4032 | Val Acc: 0.9180
Epoch [33/50] | Train Loss: 0.4356 | Val Loss: 0.4031 | Val Acc: 0.9180
Epoch [34/50] | Train Loss: 0.4322 | Val Loss: 0.4016 | Val Acc: 0.9180
Epoch [35/50] | Train Loss: 0.4295 | Val Loss: 0.3995 | Val Acc: 0.9180
Epoch [36/50] | Train Loss: 0.4300 | Val Loss: 0.3933 | Val Acc: 0.9180
Epoch [37/50] | Train Loss: 0.4288 | Val Loss: 0.3901 | Val Acc: 0.9180
Epoch [38/50] | Train Loss: 0.4245 | Val Loss: 0.3901 | Val Acc: 0.9180
Epoch [39/50] | Train Loss: 0.4214 | Val Loss: 0.3929 | Val Acc: 0.9180
Epoch [40/50] | Train Loss: 0.4185 | Val Loss: 0.3909 | Val Acc: 0.9180
Epoch [41/50] | Train Loss: 0.4164 | Val Loss: 0.3876 | Val Acc: 0.9180
Epoch [42/50] | Train Loss: 0.4144 | Val Loss: 0.3836 | Val Acc: 0.9180
Epoch [43/50] | Train Loss: 0.4126 | Val Loss: 0.3819 | Val Acc: 0.9180
Epoch [44/50] | Train Loss: 0.4109 | Val Loss: 0.3811 | Val Acc: 0.9180
Epoch [45/50] | Train Loss: 0.4092 | Val Loss: 0.3797 | Val Acc: 0.9180
Epoch [46/50] | Train Loss: 0.4074 | Val Loss: 0.3765 | Val Acc: 0.9180
Epoch [47/50] | Train Loss: 0.4059 | Val Loss: 0.3747 | Val Acc: 0.9180
Epoch [48/50] | Train Loss: 0.4041 | Val Loss: 0.3739 | Val Acc: 0.9180
Epoch [49/50] | Train Loss: 0.4022 | Val Loss: 0.3716 | Val Acc: 0.9180
Epoch [50/50] | Train Loss: 0.4005 | Val Loss: 0.3702 | Val Acc: 0.9180

Confusion Matrix:
[[25  3]
 [ 2 31]]

Classification Report:
              precision    recall  f1-score    support
0.0            0.925926  0.892857  0.909091  28.000000
1.0            0.911765  0.939394  0.925373  33.000000
accuracy       0.918033  0.918033  0.918033   0.918033
macro avg      0.918845  0.916126  0.917232  61.000000
weighted avg   0.918265  0.918033  0.917899  61.000000

================ FOLD 2 / 5 ================
Epoch [1/50] | Train Loss: 0.6335 | Val Loss: 0.6274 | Val Acc: 0.4590
Epoch [2/50] | Train Loss: 0.6182 | Val Loss: 0.6003 | Val Acc: 0.4590
Epoch [3/50] | Train Loss: 0.5955 | Val Loss: 0.5836 | Val Acc: 0.4590
Epoch [4/50] | Train Loss: 0.5845 | Val Loss: 0.5752 | Val Acc: 0.4590
Epoch [5/50] | Train Loss: 0.5767 | Val Loss: 0.5690 | Val Acc: 0.4590
Epoch [6/50] | Train Loss: 0.5710 | Val Loss: 0.5649 | Val Acc: 0.4590
Epoch [7/50] | Train Loss: 0.5650 | Val Loss: 0.5601 | Val Acc: 0.4590
Epoch [8/50] | Train Loss: 0.5589 | Val Loss: 0.5550 | Val Acc: 0.8852
Epoch [9/50] | Train Loss: 0.5537 | Val Loss: 0.5491 | Val Acc: 0.8689
Epoch [10/50] | Train Loss: 0.5490 | Val Loss: 0.5453 | Val Acc: 0.8525
Epoch [11/50] | Train Loss: 0.5445 | Val Loss: 0.5411 | Val Acc: 0.8689
Epoch [12/50] | Train Loss: 0.5399 | Val Loss: 0.5385 | Val Acc: 0.8689
Epoch [13/50] | Train Loss: 0.5390 | Val Loss: 0.5386 | Val Acc: 0.8852
Epoch [14/50] | Train Loss: 0.5334 | Val Loss: 0.5324 | Val Acc: 0.8852
Epoch [15/50] | Train Loss: 0.5281 | Val Loss: 0.5305 | Val Acc: 0.8525
Epoch [16/50] | Train Loss: 0.5239 | Val Loss: 0.5257 | Val Acc: 0.9016
Epoch [17/50] | Train Loss: 0.5204 | Val Loss: 0.5247 | Val Acc: 0.8525
Epoch [18/50] | Train Loss: 0.5161 | Val Loss: 0.5178 | Val Acc: 0.9016
Epoch [19/50] | Train Loss: 0.5128 | Val Loss: 0.5180 | Val Acc: 0.8689
Epoch [20/50] | Train Loss: 0.5087 | Val Loss: 0.5156 | Val Acc: 0.8689
Epoch [21/50] | Train Loss: 0.5052 | Val Loss: 0.5127 | Val Acc: 0.8689
Epoch [22/50] | Train Loss: 0.5022 | Val Loss: 0.5088 | Val Acc: 0.8689
Epoch [23/50] | Train Loss: 0.4983 | Val Loss: 0.5065 | Val Acc: 0.8689
Epoch [24/50] | Train Loss: 0.4952 | Val Loss: 0.5034 | Val Acc: 0.8689
Epoch [25/50] | Train Loss: 0.4919 | Val Loss: 0.5009 | Val Acc: 0.8689
Epoch [26/50] | Train Loss: 0.4889 | Val Loss: 0.4971 | Val Acc: 0.8689
Epoch [27/50] | Train Loss: 0.4859 | Val Loss: 0.4980 | Val Acc: 0.8525
Epoch [28/50] | Train Loss: 0.4843 | Val Loss: 0.4909 | Val Acc: 0.8689
Epoch [29/50] | Train Loss: 0.4821 | Val Loss: 0.4955 | Val Acc: 0.8525
Epoch [30/50] | Train Loss: 0.4772 | Val Loss: 0.4868 | Val Acc: 0.8689
Epoch [31/50] | Train Loss: 0.4750 | Val Loss: 0.4871 | Val Acc: 0.8525
Epoch [32/50] | Train Loss: 0.4730 | Val Loss: 0.4828 | Val Acc: 0.8689
Epoch [33/50] | Train Loss: 0.4686 | Val Loss: 0.4809 | Val Acc: 0.8689
Epoch [34/50] | Train Loss: 0.4659 | Val Loss: 0.4777 | Val Acc: 0.8689
Epoch [35/50] | Train Loss: 0.4632 | Val Loss: 0.4752 | Val Acc: 0.8689
Epoch [36/50] | Train Loss: 0.4606 | Val Loss: 0.4731 | Val Acc: 0.8689
Epoch [37/50] | Train Loss: 0.4581 | Val Loss: 0.4707 | Val Acc: 0.8689
Epoch [38/50] | Train Loss: 0.4556 | Val Loss: 0.4687 | Val Acc: 0.8689
Epoch [39/50] | Train Loss: 0.4532 | Val Loss: 0.4664 | Val Acc: 0.8689
Epoch [40/50] | Train Loss: 0.4508 | Val Loss: 0.4642 | Val Acc: 0.8689
Epoch [41/50] | Train Loss: 0.4485 | Val Loss: 0.4621 | Val Acc: 0.8689
Epoch [42/50] | Train Loss: 0.4461 | Val Loss: 0.4600 | Val Acc: 0.8689
Epoch [43/50] | Train Loss: 0.4435 | Val Loss: 0.4581 | Val Acc: 0.8689
Epoch [44/50] | Train Loss: 0.4413 | Val Loss: 0.4560 | Val Acc: 0.8689
Epoch [45/50] | Train Loss: 0.4391 | Val Loss: 0.4540 | Val Acc: 0.8689
Epoch [46/50] | Train Loss: 0.4369 | Val Loss: 0.4523 | Val Acc: 0.8689
Epoch [47/50] | Train Loss: 0.4346 | Val Loss: 0.4503 | Val Acc: 0.8689
Epoch [48/50] | Train Loss: 0.4324 | Val Loss: 0.4484 | Val Acc: 0.8689
Epoch [49/50] | Train Loss: 0.4305 | Val Loss: 0.4467 | Val Acc: 0.8689
Epoch [50/50] | Train Loss: 0.4282 | Val Loss: 0.4448 | Val Acc: 0.8689

Confusion Matrix:
[[22  6]
 [ 2 31]]

Classification Report:
              precision    recall  f1-score    support
0.0            0.916667  0.785714  0.846154  28.000000
1.0            0.837838  0.939394  0.885714  33.000000
accuracy       0.868852  0.868852  0.868852   0.868852
macro avg      0.877252  0.862554  0.865934  61.000000
weighted avg   0.874022  0.868852  0.867555  61.000000

================ FOLD 3 / 5 ================
Epoch [1/50] | Train Loss: 0.6652 | Val Loss: 0.6427 | Val Acc: 0.4590
Epoch [2/50] | Train Loss: 0.6085 | Val Loss: 0.5966 | Val Acc: 0.4590
Epoch [3/50] | Train Loss: 0.5712 | Val Loss: 0.5855 | Val Acc: 0.4590
Epoch [4/50] | Train Loss: 0.5544 | Val Loss: 0.5841 | Val Acc: 0.4590
Epoch [5/50] | Train Loss: 0.5442 | Val Loss: 0.5786 | Val Acc: 0.4590
Epoch [6/50] | Train Loss: 0.5349 | Val Loss: 0.5769 | Val Acc: 0.4590
Epoch [7/50] | Train Loss: 0.5263 | Val Loss: 0.5723 | Val Acc: 0.4590
Epoch [8/50] | Train Loss: 0.5193 | Val Loss: 0.5658 | Val Acc: 0.4590
Epoch [9/50] | Train Loss: 0.5134 | Val Loss: 0.5616 | Val Acc: 0.4590
Epoch [10/50] | Train Loss: 0.5071 | Val Loss: 0.5589 | Val Acc: 0.4590
Epoch [11/50] | Train Loss: 0.5021 | Val Loss: 0.5559 | Val Acc: 0.4590
Epoch [12/50] | Train Loss: 0.4966 | Val Loss: 0.5511 | Val Acc: 0.4590
Epoch [13/50] | Train Loss: 0.4919 | Val Loss: 0.5491 | Val Acc: 0.4590
Epoch [14/50] | Train Loss: 0.4871 | Val Loss: 0.5459 | Val Acc: 0.4590
Epoch [15/50] | Train Loss: 0.4823 | Val Loss: 0.5438 | Val Acc: 0.7541
Epoch [16/50] | Train Loss: 0.4782 | Val Loss: 0.5435 | Val Acc: 0.7705
Epoch [17/50] | Train Loss: 0.4745 | Val Loss: 0.5386 | Val Acc: 0.7705
Epoch [18/50] | Train Loss: 0.4698 | Val Loss: 0.5393 | Val Acc: 0.7705
Epoch [19/50] | Train Loss: 0.4652 | Val Loss: 0.5387 | Val Acc: 0.7705
Epoch [20/50] | Train Loss: 0.4620 | Val Loss: 0.5367 | Val Acc: 0.7705
Epoch [21/50] | Train Loss: 0.4569 | Val Loss: 0.5352 | Val Acc: 0.7705
Epoch [22/50] | Train Loss: 0.4531 | Val Loss: 0.5337 | Val Acc: 0.7705
Epoch [23/50] | Train Loss: 0.4491 | Val Loss: 0.5325 | Val Acc: 0.7705
Epoch [24/50] | Train Loss: 0.4454 | Val Loss: 0.5310 | Val Acc: 0.7705
Epoch [25/50] | Train Loss: 0.4414 | Val Loss: 0.5297 | Val Acc: 0.7705
Epoch [26/50] | Train Loss: 0.4425 | Val Loss: 0.5273 | Val Acc: 0.7705
Epoch [27/50] | Train Loss: 0.4353 | Val Loss: 0.5247 | Val Acc: 0.7705
Epoch [28/50] | Train Loss: 0.4313 | Val Loss: 0.5241 | Val Acc: 0.7705
Epoch [29/50] | Train Loss: 0.4276 | Val Loss: 0.5230 | Val Acc: 0.7705
Epoch [30/50] | Train Loss: 0.4233 | Val Loss: 0.5213 | Val Acc: 0.7705
Epoch [31/50] | Train Loss: 0.4201 | Val Loss: 0.5200 | Val Acc: 0.7705
Epoch [32/50] | Train Loss: 0.4163 | Val Loss: 0.5185 | Val Acc: 0.7705
Epoch [33/50] | Train Loss: 0.4132 | Val Loss: 0.5172 | Val Acc: 0.7705
Epoch [34/50] | Train Loss: 0.4094 | Val Loss: 0.5158 | Val Acc: 0.7705
Epoch [35/50] | Train Loss: 0.4063 | Val Loss: 0.5146 | Val Acc: 0.7705
Epoch [36/50] | Train Loss: 0.4033 | Val Loss: 0.5133 | Val Acc: 0.7705
Epoch [37/50] | Train Loss: 0.4006 | Val Loss: 0.5122 | Val Acc: 0.7705
Epoch [38/50] | Train Loss: 0.3972 | Val Loss: 0.5112 | Val Acc: 0.7705
Epoch [39/50] | Train Loss: 0.3943 | Val Loss: 0.5101 | Val Acc: 0.7705
Epoch [40/50] | Train Loss: 0.3920 | Val Loss: 0.5093 | Val Acc: 0.7705
Epoch [41/50] | Train Loss: 0.3885 | Val Loss: 0.5083 | Val Acc: 0.7705
Epoch [42/50] | Train Loss: 0.3857 | Val Loss: 0.5075 | Val Acc: 0.7705
Epoch [43/50] | Train Loss: 0.3840 | Val Loss: 0.5064 | Val Acc: 0.7705
Epoch [44/50] | Train Loss: 0.3806 | Val Loss: 0.5055 | Val Acc: 0.7705
Epoch [45/50] | Train Loss: 0.3797 | Val Loss: 0.5046 | Val Acc: 0.7705
Epoch [46/50] | Train Loss: 0.3751 | Val Loss: 0.5042 | Val Acc: 0.7705
Epoch [47/50] | Train Loss: 0.3733 | Val Loss: 0.5034 | Val Acc: 0.7705
Epoch [48/50] | Train Loss: 0.3710 | Val Loss: 0.5026 | Val Acc: 0.7705
Epoch [49/50] | Train Loss: 0.3689 | Val Loss: 0.5019 | Val Acc: 0.7705
Epoch [50/50] | Train Loss: 0.3642 | Val Loss: 0.5011 | Val Acc: 0.7705

Confusion Matrix:
[[19  9]
 [ 5 28]]

Classification Report:
              precision    recall  f1-score    support
0.0            0.791667  0.678571  0.730769  28.000000
1.0            0.756757  0.848485  0.800000  33.000000
accuracy       0.770492  0.770492  0.770492   0.770492
macro avg      0.774212  0.763528  0.765385  61.000000
weighted avg   0.772781  0.770492  0.768222  61.000000

================ FOLD 4 / 5 ================
Epoch [1/50] | Train Loss: 0.6838 | Val Loss: 0.6830 | Val Acc: 0.4500
Epoch [2/50] | Train Loss: 0.6761 | Val Loss: 0.6755 | Val Acc: 0.4500
Epoch [3/50] | Train Loss: 0.6697 | Val Loss: 0.6701 | Val Acc: 0.4500
Epoch [4/50] | Train Loss: 0.6641 | Val Loss: 0.6649 | Val Acc: 0.4500
Epoch [5/50] | Train Loss: 0.6585 | Val Loss: 0.6596 | Val Acc: 0.4500
Epoch [6/50] | Train Loss: 0.6531 | Val Loss: 0.6541 | Val Acc: 0.4500
Epoch [7/50] | Train Loss: 0.6473 | Val Loss: 0.6494 | Val Acc: 0.4500
Epoch [8/50] | Train Loss: 0.6417 | Val Loss: 0.6444 | Val Acc: 0.4500
Epoch [9/50] | Train Loss: 0.6357 | Val Loss: 0.6386 | Val Acc: 0.4500
Epoch [10/50] | Train Loss: 0.6302 | Val Loss: 0.6333 | Val Acc: 0.4500
Epoch [11/50] | Train Loss: 0.6246 | Val Loss: 0.6273 | Val Acc: 0.4500
Epoch [12/50] | Train Loss: 0.6193 | Val Loss: 0.6232 | Val Acc: 0.4500
Epoch [13/50] | Train Loss: 0.6141 | Val Loss: 0.6189 | Val Acc: 0.4500
Epoch [14/50] | Train Loss: 0.6082 | Val Loss: 0.6136 | Val Acc: 0.4500
Epoch [15/50] | Train Loss: 0.6028 | Val Loss: 0.6099 | Val Acc: 0.4500
Epoch [16/50] | Train Loss: 0.5972 | Val Loss: 0.6081 | Val Acc: 0.4500
Epoch [17/50] | Train Loss: 0.5917 | Val Loss: 0.6027 | Val Acc: 0.4500
Epoch [18/50] | Train Loss: 0.5873 | Val Loss: 0.5987 | Val Acc: 0.4500
Epoch [19/50] | Train Loss: 0.5825 | Val Loss: 0.5939 | Val Acc: 0.4500
Epoch [20/50] | Train Loss: 0.5781 | Val Loss: 0.5941 | Val Acc: 0.4500
Epoch [21/50] | Train Loss: 0.5730 | Val Loss: 0.5894 | Val Acc: 0.4500
Epoch [22/50] | Train Loss: 0.5680 | Val Loss: 0.5855 | Val Acc: 0.4500
Epoch [23/50] | Train Loss: 0.5635 | Val Loss: 0.5818 | Val Acc: 0.4500
Epoch [24/50] | Train Loss: 0.5590 | Val Loss: 0.5771 | Val Acc: 0.4500
Epoch [25/50] | Train Loss: 0.5547 | Val Loss: 0.5734 | Val Acc: 0.4500
Epoch [26/50] | Train Loss: 0.5504 | Val Loss: 0.5700 | Val Acc: 0.4500
Epoch [27/50] | Train Loss: 0.5463 | Val Loss: 0.5667 | Val Acc: 0.4500
Epoch [28/50] | Train Loss: 0.5425 | Val Loss: 0.5636 | Val Acc: 0.4500
Epoch [29/50] | Train Loss: 0.5383 | Val Loss: 0.5605 | Val Acc: 0.4500
Epoch [30/50] | Train Loss: 0.5344 | Val Loss: 0.5568 | Val Acc: 0.4500
Epoch [31/50] | Train Loss: 0.5304 | Val Loss: 0.5538 | Val Acc: 0.4500
Epoch [32/50] | Train Loss: 0.5262 | Val Loss: 0.5510 | Val Acc: 0.4500
Epoch [33/50] | Train Loss: 0.5225 | Val Loss: 0.5475 | Val Acc: 0.4500
Epoch [34/50] | Train Loss: 0.5188 | Val Loss: 0.5451 | Val Acc: 0.4500
Epoch [35/50] | Train Loss: 0.5147 | Val Loss: 0.5423 | Val Acc: 0.4500
Epoch [36/50] | Train Loss: 0.5115 | Val Loss: 0.5394 | Val Acc: 0.4500
Epoch [37/50] | Train Loss: 0.5074 | Val Loss: 0.5368 | Val Acc: 0.8000
Epoch [38/50] | Train Loss: 0.5044 | Val Loss: 0.5343 | Val Acc: 0.8000
Epoch [39/50] | Train Loss: 0.5004 | Val Loss: 0.5319 | Val Acc: 0.8000
Epoch [40/50] | Train Loss: 0.4967 | Val Loss: 0.5295 | Val Acc: 0.8000
Epoch [41/50] | Train Loss: 0.4936 | Val Loss: 0.5267 | Val Acc: 0.8000
Epoch [42/50] | Train Loss: 0.4908 | Val Loss: 0.5250 | Val Acc: 0.8000
Epoch [43/50] | Train Loss: 0.4868 | Val Loss: 0.5218 | Val Acc: 0.8167
Epoch [44/50] | Train Loss: 0.4829 | Val Loss: 0.5198 | Val Acc: 0.8000
Epoch [45/50] | Train Loss: 0.4800 | Val Loss: 0.5171 | Val Acc: 0.8167
Epoch [46/50] | Train Loss: 0.4765 | Val Loss: 0.5145 | Val Acc: 0.8167
Epoch [47/50] | Train Loss: 0.4732 | Val Loss: 0.5126 | Val Acc: 0.8167
Epoch [48/50] | Train Loss: 0.4709 | Val Loss: 0.5106 | Val Acc: 0.8167
Epoch [49/50] | Train Loss: 0.4667 | Val Loss: 0.5091 | Val Acc: 0.8000
Epoch [50/50] | Train Loss: 0.4651 | Val Loss: 0.5060 | Val Acc: 0.8167

Confusion Matrix:
[[21  6]
 [ 5 28]]

Classification Report:
              precision    recall  f1-score    support
0.0            0.807692  0.777778  0.792453  27.000000
1.0            0.823529  0.848485  0.835821  33.000000
accuracy       0.816667  0.816667  0.816667   0.816667
macro avg      0.815611  0.813131  0.814137  60.000000
weighted avg   0.816403  0.816667  0.816305  60.000000

================ FOLD 5 / 5 ================
Epoch [1/50] | Train Loss: 0.7224 | Val Loss: 0.7102 | Val Acc: 0.5500
Epoch [2/50] | Train Loss: 0.7080 | Val Loss: 0.6930 | Val Acc: 0.5500
Epoch [3/50] | Train Loss: 0.6845 | Val Loss: 0.6776 | Val Acc: 0.5500
Epoch [4/50] | Train Loss: 0.6714 | Val Loss: 0.6694 | Val Acc: 0.5500
Epoch [5/50] | Train Loss: 0.6634 | Val Loss: 0.6624 | Val Acc: 0.5500
Epoch [6/50] | Train Loss: 0.6556 | Val Loss: 0.6558 | Val Acc: 0.5500
Epoch [7/50] | Train Loss: 0.6493 | Val Loss: 0.6478 | Val Acc: 0.5500
Epoch [8/50] | Train Loss: 0.6406 | Val Loss: 0.6429 | Val Acc: 0.5500
Epoch [9/50] | Train Loss: 0.6338 | Val Loss: 0.6399 | Val Acc: 0.5500
Epoch [10/50] | Train Loss: 0.6275 | Val Loss: 0.6351 | Val Acc: 0.5500
Epoch [11/50] | Train Loss: 0.6215 | Val Loss: 0.6308 | Val Acc: 0.5500
Epoch [12/50] | Train Loss: 0.6157 | Val Loss: 0.6261 | Val Acc: 0.5500
Epoch [13/50] | Train Loss: 0.6107 | Val Loss: 0.6219 | Val Acc: 0.5500
Epoch [14/50] | Train Loss: 0.6049 | Val Loss: 0.6173 | Val Acc: 0.5500
Epoch [15/50] | Train Loss: 0.5990 | Val Loss: 0.6139 | Val Acc: 0.5500
Epoch [16/50] | Train Loss: 0.5944 | Val Loss: 0.6095 | Val Acc: 0.5500
Epoch [17/50] | Train Loss: 0.5895 | Val Loss: 0.6043 | Val Acc: 0.5500
Epoch [18/50] | Train Loss: 0.5833 | Val Loss: 0.6006 | Val Acc: 0.5500
Epoch [19/50] | Train Loss: 0.5784 | Val Loss: 0.5984 | Val Acc: 0.5500
Epoch [20/50] | Train Loss: 0.5734 | Val Loss: 0.5949 | Val Acc: 0.5500
Epoch [21/50] | Train Loss: 0.5687 | Val Loss: 0.5911 | Val Acc: 0.5500
Epoch [22/50] | Train Loss: 0.5642 | Val Loss: 0.5874 | Val Acc: 0.5500
Epoch [23/50] | Train Loss: 0.5595 | Val Loss: 0.5846 | Val Acc: 0.5500
Epoch [24/50] | Train Loss: 0.5552 | Val Loss: 0.5816 | Val Acc: 0.5500
Epoch [25/50] | Train Loss: 0.5509 | Val Loss: 0.5778 | Val Acc: 0.5500
Epoch [26/50] | Train Loss: 0.5463 | Val Loss: 0.5738 | Val Acc: 0.5500
Epoch [27/50] | Train Loss: 0.5426 | Val Loss: 0.5718 | Val Acc: 0.5500
Epoch [28/50] | Train Loss: 0.5383 | Val Loss: 0.5675 | Val Acc: 0.5500
Epoch [29/50] | Train Loss: 0.5386 | Val Loss: 0.5647 | Val Acc: 0.5500
Epoch [30/50] | Train Loss: 0.5321 | Val Loss: 0.5631 | Val Acc: 0.5500
Epoch [31/50] | Train Loss: 0.5265 | Val Loss: 0.5605 | Val Acc: 0.5500
Epoch [32/50] | Train Loss: 0.5221 | Val Loss: 0.5576 | Val Acc: 0.5500
Epoch [33/50] | Train Loss: 0.5186 | Val Loss: 0.5525 | Val Acc: 0.5500
Epoch [34/50] | Train Loss: 0.5185 | Val Loss: 0.5503 | Val Acc: 0.5500
Epoch [35/50] | Train Loss: 0.5177 | Val Loss: 0.5442 | Val Acc: 0.5500
Epoch [36/50] | Train Loss: 0.5109 | Val Loss: 0.5444 | Val Acc: 0.5500
Epoch [37/50] | Train Loss: 0.5049 | Val Loss: 0.5372 | Val Acc: 0.5500
Epoch [38/50] | Train Loss: 0.5008 | Val Loss: 0.5383 | Val Acc: 0.8167
Epoch [39/50] | Train Loss: 0.4971 | Val Loss: 0.5374 | Val Acc: 0.8333
Epoch [40/50] | Train Loss: 0.4939 | Val Loss: 0.5359 | Val Acc: 0.8333
Epoch [41/50] | Train Loss: 0.4909 | Val Loss: 0.5337 | Val Acc: 0.8333
Epoch [42/50] | Train Loss: 0.4887 | Val Loss: 0.5319 | Val Acc: 0.8167
Epoch [43/50] | Train Loss: 0.4850 | Val Loss: 0.5300 | Val Acc: 0.8167
Epoch [44/50] | Train Loss: 0.4803 | Val Loss: 0.5282 | Val Acc: 0.8167
Epoch [45/50] | Train Loss: 0.4763 | Val Loss: 0.5259 | Val Acc: 0.8167
Epoch [46/50] | Train Loss: 0.4775 | Val Loss: 0.5245 | Val Acc: 0.8167
Epoch [47/50] | Train Loss: 0.4715 | Val Loss: 0.5172 | Val Acc: 0.8333
Epoch [48/50] | Train Loss: 0.4707 | Val Loss: 0.5199 | Val Acc: 0.8167
Epoch [49/50] | Train Loss: 0.4658 | Val Loss: 0.5177 | Val Acc: 0.8167
Epoch [50/50] | Train Loss: 0.4618 | Val Loss: 0.5160 | Val Acc: 0.8167

Confusion Matrix:
[[21  6]
 [ 5 28]]

Classification Report:
              precision    recall  f1-score    support
0.0            0.807692  0.777778  0.792453  27.000000
1.0            0.823529  0.848485  0.835821  33.000000
accuracy       0.816667  0.816667  0.816667   0.816667
macro avg      0.815611  0.813131  0.814137  60.000000
weighted avg   0.816403  0.816667  0.816305  60.000000

================ AVERAGE RESULTS (5-FOLD) ================

Summed Confusion Matrix:
[[108  30]
 [ 19 146]]

Average Classification Report (metrics only):
              precision    recall  f1-score
0.0            0.849929  0.782540  0.814184
1.0            0.830684  0.884848  0.856546
macro avg      0.840306  0.833694  0.835365
weighted avg   0.839575  0.838142  0.837257

==== CROSS-VALIDATION COMPLETE ====

(.venv) C:\Users\ME\Desktop\hybrid-fuzzy-ensemble>
