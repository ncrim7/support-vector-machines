# DESTEK VEKTÖR MAKİNELERİ (SVM)

Bu belge, **Destek Vektör Makineleri (SVM)** algoritmasının detaylı bir incelemesini içermektedir. Bu çalışma, algoritmanın teorik temelleri, matematiksel açıklamaları, farklı çekirdek fonksiyonlarının kullanımı ve uygulama alanları hakkında kapsamlı bilgiler sunmaktadır.

## İçindekiler

1. [DESTEK VEKTÖR MAKİNELERİ (SVM)](#1-destek-vektör-makineleri-svm)
   - [Giriş](#11-giriş)
     - [Algoritmanın Tarihçesi ve Keşfi](#111-algoritmanın-tarihçesi-ve-keşfi)
     - [Nerelerde ve Neden Kullanılır](#112-nerelerde-ve-neden-kullanılır)
     - [Gerçek Dünya Örnekleri](#113-gerçek-dünya-örnekleri)
   - [Teorik Temel](#12-teorik-temel)
     - [Algoritmanın Matematiksel ve Teorik Açıklaması](#121-algoritmanın-matematiksel-ve-teorik-açıklaması)
     - [Doğrusal SVM](#122-doğrusal-svm)
     - [Doğrusal Olmayan SVM](#123-doğrusal-olmayan-svm)
     - [Optimizasyon Problemi](#124-optimizasyon-problemi)
     - [Çekirdek Hilesi (Kernel Trick)](#125-çekirdek-hilesi-kernel-trick)
     - [Polinom Çekirdek](#126-polinom-çekirdek)
     - [RBF Çekirdek](#127-rbf-çekirdek)
     - [Algoritmanın Avantajları ve Sınırlamaları](#128-algoritmanın-avantajları-ve-sınırlamaları)
   - [Algoritmanın Uygulama Alanları](#13-algoritmanın-uygulama-alanları)
     - [Algoritmanın Açıklanması](#131-algoritmanın-açıklanması)
     - [Algoritma](#132-algoritma)
     - [Kullanım Alanları](#133-kullanım-alanları)
   - [Performans Analizi](#14-performans-analizi)
     - [Uzay ve Zaman Karmaşıklığı](#141-uzay-ve-zaman-karmaşıklığı)
     - [Optimizasyon Seçenekleri](#142-optimizasyon-seçenekleri)
   - [Çalışma Soruları ve Egzersizler](#15-çalışma-soruları-ve-egzersizler)
   - [Algoritma Özeti](#16-algoritma-özeti)

## 1. DESTEK VEKTÖR MAKİNELERİ (SVM)

### 1.1 Giriş

Destek Vektör Makineleri (SVM), gözetimli öğrenme yöntemlerinden biridir ve sınıflandırma ve regresyon problemlerinde kullanılır. Bu algoritma, özellikle yüksek doğruluk ve sınıflandırma güvenilirliği sağlamak amacıyla tercih edilmektedir.

#### 1.1.1 Algoritmanın Tarihçesi ve Keşfi
Destek Vektör Makineleri (SVM), 1960'larda Vladimir Vapnik ve Alexey Chervonenkis tarafından 
geliştirilmeye başlanmış bir makine öğrenmesi algoritmasıdır. Fakat SVM'lerin gerçek potansiyeli, 
1990'larda Bernhard Boser, Isabelle Guyon ve Vladimir Vapnik tarafından tanıtılan çekirdek hilesiyle 
(kernel trick) ortaya çıkmıştır. Bu yenilik, SVM'lerin doğrusal olmayan problemlere uygulanabilmesini 
sağlayarak popülerliğini büyük ölçüde artırmıştır. 

#### 1.1.2 Nerelerde ve Neden Kullanılır
Destek Vektör Makineleri (SVM), sınıflandırma ve regresyon problemlerinde oldukça etkili bir 
yöntemdir. Özellikle küçük ve orta ölçekli veri setlerinde ve yüksek boyutlu verilerde başarılı sonuçlar 
verir. İşte Destek Vektör Makinelerinin tercih edilme nedenlerinden bazıları: 
Küçük veri setlerinde üstün performans: Destek Vektör Makineleri (SVM), az sayıda veri noktasıyla 
bile çok iyi derecede genelleme yeteneği gösterir. 
Yüksek boyutlu verilerle etkin çalışma: Çekirdek hilesi sayesinde Destek Vektör Makineleri (SVM), 
yüksek boyutlu verileri düşük boyutlu bir uzaya dönüştürerek karmaşık problemleri başarılı bir şekilde 
çözebilir. 
Daha az ayarlanabilir parametre: Destek Vektör Makineleri (SVM), diğer algoritmalara göre daha az 
ayarlanabilir parametreye sahiptir, bu da modelin daha hızlı ve kolay bir şekilde eğitilmesini sağlar. 

#### 1.1.3 Gerçek Dünya Örnekleri
Destek Vektör Makineleri (SVM), görüntü tanıma, el yazısı tanıma, yüz tanıma, nesne tanıma gibi 
birçok alanda etkili bir şekilde kullanılmaktadır. Ayrıca, spam e-posta tespiti, duygu analizi ve metinleri 
kategorize etme gibi metin sınıflandırma işlemlerinde de başarılıdır. Biyomedikal alanda kanser teşhisi 
ve gen ifadesi analizi gibi karmaşık problemleri çözmede büyük rol oynar. Finans sektöründe 
dolandırıcılık tespiti ve hisse senedi fiyat tahmini gibi uygulamalarda da kullanılır. Diğer alanlarda ise 
ses tanıma ve protein yapısı tahmini gibi sorunları çözmede etkilidir. SVM'nin bu kadar popüler 
olmasının nedeni, farklı alanlardaki karmaşık problemleri çözme yeteneğidir. Örneğin, bir görüntü 
tanıma sisteminde SVM, bir görüntüyü oluşturan pikselleri yüksek boyutlu bir vektör olarak temsil eder 
ve bu vektörleri farklı sınıflara (örneğin, kedi, köpek, araba) ait örneklerle karşılaştırır. Böylece, SVM, 
verilen bir görüntünün hangi sınıfa ait olduğunu doğru bir şekilde tahmin edebilir. 
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/1_fpDngO6lM5pDeIPOOezK1g_op.webp" width="auto" height="auto">
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/7364c7c7885b8652083ac6ff7de229ff.jpg" width="auto" height="auto">

### 1.2 Teorik Temel

SVM'nin matematiksel temeli, veri noktalarını doğrusal ve doğrusal olmayan sınıflara ayırmaya yönelik optimizasyon tekniklerine dayanır.

#### 1.2.1 Algoritmanın Matematiksel ve Teorik Açıklaması
Veri sınıflandırma, makine öğreniminde yaygın bir görevdir. Belirli veri noktalarının her birinin iki 
sınıftan birine ait olduğunu ve amacın yeni bir veri noktasının hangi sınıfta olacağını belirlemek 
olduğunu varsayalım. Destek Vektör Makineleri (SVM) durumunda, bir veri noktası p-boyutlu bir 
vektör (p sayısı kadar listeden oluşan) olarak görülür ve bu tür noktaları (p-1)-boyutlu bir hiper düzlemle 
ayırıp ayıramayacağımızı bilmek isteriz. Buna doğrusal sınıflandırıcı denir. Veriyi sınıflandırabilecek 
birçok hiper düzlem vardır. En iyi hiper düzlem olarak makul bir seçenek, iki sınıf arasındaki en büyük 
ayrımı veya marjı temsil eden hiper düzlemdir. Yani, her iki yandaki en yakın veri noktasına olan mesafe 
maksimize edilecek şekilde hiper düzlemi seçiyoruz. Böyle bir hiper düzlem varsa, buna maksimum 
marjlı hiper düzlem denir ve tanımladığı doğrusal sınıflandırıcıya maksimum marjlı sınıflandırıcı denir; 
ya da eşdeğer olarak, optimal stabiliteye sahip algılayıcı denir. 
Daha resmi olarak, bir Destek Vektör Makinesi (SVM), yüksek veya sonsuz boyutlu bir uzayda hiper 
düzlem veya hiper düzlemler kümesi oluşturur. Bu, sınıflandırma, regresyon veya aykırı değerlerin 
tespiti gibi diğer görevler için kullanılabilir. Sezgisel olarak, iyi bir ayrım, herhangi bir sınıfın en yakın 
eğitim verisi noktasına en büyük mesafeyi (işlevsel marj olarak adlandırılır) sahip olan hiper düzlem 
tarafından sağlanır. Genelde marj ne kadar büyük olursa, sınıflandırıcının genelleme hatası o kadar 
düşük olur. Daha düşük bir genelleme hatası, uygulayıcının aşırı öğrenim (overfitting) yaşama 
olasılığının daha düşük olduğu anlamına gelir. 
Ancak, başlangıçtaki problem sonlu boyutlu bir uzayda ifade edilse de, genellikle ayrılması gereken 
kümeler bu uzayda doğrusal olarak ayrılabilir değildir. Bu nedenle, orijinal sonlu boyutlu uzayın çok 
daha yüksek boyutlu bir uzaya dönüştürülmesi önerildi; bu şekilde ayrım daha kolay yapılabilir. 
Hesaplama yükünü makul düzeyde tutmak için, SVM şemalarında kullanılan dönüşümler, giriş veri 
vektör çiftlerinin nokta çarpımlarının, orijinal uzaydaki değişkenler cinsinden kolayca 
hesaplanabilmesini sağlamak üzere tasarlanmıştır. Bu, problem için uygun bir çekirdek fonksiyonu 
k(x,y) tanımlanarak yapılır. Daha yüksek boyutlu uzaydaki hiper düzlemler, bu uzaydaki bir vektörle 
nokta çarpımı sabit olan noktalar kümesi olarak tanımlanır. Hiper düzlemleri tanımlayan vektörler, veri 
tabanında bulunan özellik vektörlerinin görüntülerinin αi parametreleriyle lineer kombinasyonları 
olarak seçilebilir. Bu şekilde bir hiper düzlem seçildiğinde, özellik uzayındaki x noktaları, Σi.αi.k(xi,x) 
= sabit bağıntısıyla tanımlanır. k(x,y) x'den uzaklaştıkça küçülürse, toplamın her bir terimi, test 
noktasının x'e karşılık gelen veri tabanı noktasına yakınlığını ölçer. Bu şekilde, yukarıdaki çekirdeklerin 
toplamı, her bir test noktasının ayrım yapılacak iki kümeden hangisine daha yakın olduğunu ölçmek için 
kullanılabilir. Bu, herhangi bir hiper düzleme dönüştürülen x noktalarının setinin oldukça karmaşık 
olabileceği ve bu nedenle, orijinal uzayda konveks olmayan kümeler arasında çok daha karmaşık 
ayrımlar yapılmasına olanak tanıdığı anlamına gelir.

#### 1.2.2 Doğrusal SVM
Bize (𝒙𝟏,𝒚𝟏),…,(𝒙𝒏,𝒚𝒏) formundaki n noktadan oluşan bir eğitim veri kümesi verilir; burada y, ya 1 
ya da -1’dir, her biri x noktasının ait olduğu sınıfı belirtir. Her x, p boyutlu bir gerçek vektördür. 𝒚𝒊 = 1 
olan x nokta grubunu, 𝒚𝒊 = 1 olan nokta grubundan ayıran "maksimum kenar hiperdüzlemini" bulmak 
istiyoruz; hiperdüzlem ve her iki gruptan en yakın x noktası maksimize edilir. Herhangi bir hiperdüzlem, 
�
�𝑻𝒙−𝒃=𝟎 'ı sağlayan x noktaları kümesi olarak yazılabilir; burada w, hiperdüzlemin (zorunlu olarak 
normalleştirilmiş olması gerekmez) normal vektörüdür. Bu, w’nin mutlaka bir birim vektör olmaması 
dışında Hesse normal formuna çok benzer. Parametre, hiperdüzlemin w normal vektörü boyunca 
orijinden uzaklığını belirler. Uyarı: Konuyla ilgili literatürün çoğu, iki sınıftan örneklerle eğitilmiş bir 
SVM için Maksimum kenar boşluğu hiperdüzlemi ve kenar boşlukları olacak şekilde önyargıyı tanımlar. 
Kenardaki örneklere destek vektörleri denir. 𝒘𝑻𝒙 + 𝒃 = 𝟎 
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/SVM_margin.png" width="auto" height="auto">

#### 1.2.3 Doğrusal Olmayan SVM
Destek Vektör Makineleri (SVM), verileri optimal bir hiper düzlem ile ayırmayı amaçlayan güçlü 
makine öğrenmesi algoritmalarıdır, ancak tüm veri kümeleri doğrusal olarak ayrılabilir olmayabilir; bu 
durumda doğrusal olmayan SVM sınıflandırıcılar devreye girer. Doğrusal olmayan veri kümeleri, bir 
çizgi veya düzlem ile ayrılamayan, örneğin bir dairenin içindeki ve dışındaki noktaları ayıran verilerdir 
ve bu tür veri kümeleri için klasik SVM yöntemleri yetersiz kalır. Doğrusal olmayan SVM sınıflandırıcı, 
çekirdek hilesi kullanarak bu tür verileri etkili bir şekilde sınıflandırabilir ve farklı çekirdek 
fonksiyonları ve hiperparametre ayarları ile çeşitli problemlere uyum sağlar. Bununla birlikte, büyük 
veri setlerinde performans düşebilir ve hiperparametre seçimi dikkatle yapılmalıdır; bu algoritmanın 
güçlü yönleri arasında küçük veri setlerinde iyi performans göstermesi, yüksek boyutlu verilerle 
çalışabilmesi ve az sayıda ayarlanabilir parametre gerektirmesi yer alırken, sınırlamaları arasında büyük 
veri setlerinde yavaş çalışması, karmaşık doğrusal olmayan problemlerde performans düşüşü ve aykırı 
değerlere duyarlılık bulunmaktadır. Sonuç olarak, SVM, çeşitli alanlarda başarıyla kullanılan güçlü bir 
sınıflandırma algoritmasıdır; ancak, algoritma seçimi yaparken verinin özellikleri, problemin 
karmaşıklığı ve istenen performans gibi faktörler dikkate alınmalıdır.
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/Ekran%20Al%C4%B1nt%C4%B1s%C4%B1.PNG" width="auto" height="auto">

#### 1.2.4 Optimizasyon Problemi
SVM algoritmasının başarısının merkezinde, veri noktalarını en iyi şekilde ayıracak bir hiper düzlemi 
bulmayı hedefleyen bir optimizasyon problemi vardır. Bu problemi çözmek, SVM'nin etkinliğini 
belirleyen en kritik faktörlerden biridir. SVM'deki asıl amaç, verilen bir veri kümesini en büyük marjla 
ayıran bir hiper düzlem bulmaktır. Bu, şu şekilde matematiksel bir optimizasyon problemine 
dönüştürülür: 𝑴𝒊𝒏𝒊𝒎𝒊𝒛𝒆: 𝟏 𝟐
 ⁄ ∣∣ 𝒘∣∣𝟐 𝑺𝒖𝒃𝒋𝒆𝒄𝒕 𝒕𝒐: 𝒚𝒊(𝒘𝑻 ⋅𝒙𝒊 +𝒃) ≥ 𝟏, 𝒇𝒐𝒓 𝒊 = 𝟏,...,𝒏 

Bu formülde: 
w: Hiper düzlemin normal vektörü 
b: Hiper düzlemin kaydırma miktarı 
�
�𝒊: i'inci veri noktası 
�
�𝒊: i'inci veri noktasının etiketi (1 veya -1) 
Bu problem, kısıtlı kuadratik programlama (quadratic programming) olarak bilinir. Çözümü, özel 
algoritmalar ve yazılımlar gerektirir. 

#### 1.2.5 Çekirdek Hilesi (Kernel Trick)
Çekirdek Hilesi (Kernel Trick), Destek Vektör Makineleri (SVM) algoritmasının en güçlü 
özelliklerinden biridir ve bu teknik, doğrusal olmayan ayrılabilir veri kümelerini yüksek boyutlu bir 
uzaya dönüştürerek, orada doğrusal bir ayırma yapılmasını sağlar, bu sayede daha karmaşık problemler 
çözülebilir. Gerçek dünyadaki birçok veri kümesi doğrusal bir çizgi veya hiper düzlem ile tam olarak 
ayrılamaz ve verileri yüksek boyutlu bir uzaya dönüştürerek daha karmaşık ilişkiler yakalanabilir, ancak 
yüksek boyutlu uzaylarda hesaplamaların karmaşıklığı artar. Bu nedenle, veri noktalarını yüksek 
boyutlu bir uzaya dönüştürmek yerine, bu dönüşümün sonucunda elde edilecek iç çarpımları doğrudan 
hesaplayan bir fonksiyon (çekirdek fonksiyonu) kullanılır. Farklı çekirdek fonksiyonları, farklı türdeki 
doğrusal olmayan ilişkileri yakalar; örneğin, lineer çekirdek doğrusal ayrım için kullanılırken, polinom 
çekirdek daha karmaşık, polinomsal ilişkiler için ve RBF (Radial Basis Function) çekirdek en yaygın 
kullanılan çekirdek olarak veri noktaları arasındaki benzerliği ölçer. Çekirdek hilesi, verileri açıkça 
yüksek boyutlu bir uzaya dönüştürmeye gerek kalmadan, çekirdek fonksiyonları aracılığıyla bu 
dönüşümü ima eder ve yüksek boyutlu uzaylardaki hesaplamalar yerine çekirdek matrisi adı verilen bir 
matris üzerinde işlemler yaparak hesaplama karmaşıklığını azaltır. Farklı çekirdek fonksiyonları 
sayesinde, farklı türdeki verilere uyum sağlamak mümkündür. Örneğin, RBF çekirdeği iki veri noktası 
arasındaki Öklid uzaklığına bağlı olarak bir benzerlik ölçüsü verir ve genellikle doğrusal olmayan ayrım 
problemleri için iyi sonuçlar verir. Çekirdek hilesi, SVM algoritmasının gücünü artıran önemli bir 
tekniktir, doğrusal olmayan ayrılabilir veri kümelerini başarıyla sınıflandırmamızı sağlar, farklı çekirdek 
fonksiyonları seçerek, farklı türdeki veriler için en uygun modeli elde edebiliriz, doğrusal olmayan veri 
kümelerini ayırabilir, verileri yüksek boyutlu uzaylara dönüştürerek daha karmaşık ilişkileri yakalar, 
farklı veri türlerine uyum sağlar ve çekirdek matrisi sayesinde hesaplama yükünü azaltır. 

<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/Kernel_trick_idea.svg.png" width="auto" height="auto">

#### 1.2.6 Polinom Çekirdek
ve bu fonksiyon, verileri daha yüksek boyutlu bir uzaya dönüştürerek doğrusal olmayan verilerin daha 
iyi sınıflandırılmasını sağlar. Polinom çekirdek, iki veri noktası arasındaki ilişkiyi bir polinom fonksiyonu ile ifade eder ve böylece veriler arasındaki daha karmaşık ilişkiler yakalanabilir; genel olarak 
K(x, y) = (γ <x, y> + r)^d formülü ile tanımlanır, burada K(x, y) x ve y veri noktaları arasındaki çekirdek 
değerini, γ çekirdeğin genişliğini, <x, y> x ve y vektörlerinin iç çarpımını, r sabit bir terimi ve d 
polinomun derecesini ifade eder. Polinomun derecesi olan d, polinomun karmaşıklığını belirler; d değeri 
arttıkça polinom daha karmaşık hale gelir ve daha yüksek dereceden etkileşimler yakalanabilir, ancak 
çok yüksek dereceler modelin aşırı öğrenme riskini artırabilir. Polinom çekirdekler, doğrusal olarak 
ayrılamayan verileri daha yüksek boyutlu bir uzaya dönüştürerek orada doğrusal bir hiper düzlem ile 
ayırmayı sağlar, veriler arasındaki karmaşık ve doğrusal olmayan ilişkileri yakalayabilir ve d 
parametresi sayesinde farklı derecelerdeki polinomlar kullanılarak modelin karmaşıklığı ayarlanabilir. 
Polinom çekirdek, veriler arasında polinom bir ilişki olduğu düşünüldüğünde, verilerin doğrusal 
olmayan bir yapıya sahip olduğu durumlarda ve diğer çekirdek fonksiyonlarının (örneğin RBF) iyi sonuç 
vermediği durumlarda kullanılır. Polinom çekirdek, SVM'lerde doğrusal olmayan verileri 
sınıflandırmak için güçlü bir araçtır ve veriler arasındaki polinom ilişkileri yakalayarak daha iyi bir 
model oluşturulmasını sağlar, ancak doğru parametrelerin seçilmesi ve aşırı öğrenme riskine dikkat 
edilmesi önemlidir. Bu çekirdek, verileri daha yüksek boyutlu bir uzaya dönüştürerek doğrusal olmayan 
ilişkileri yakalar ve d parametresi, polinomun derecesini belirler; doğru parametre seçimi, modelin 
başarısı için kritik öneme sahiptir. 

<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/Ekran%20Al%C4%B1nt%C4%B1s%C4%B12.PNG" width="auto" height="auto">

#### 1.2.7 RBF Çekirdek
Radyal Temel Fonksiyon (RBF) çekirdek, Destek Vektör Makinelerinde (SVM) sıkça kullanılan ve 
doğrusal olmayan verilerin sınıflandırılmasında etkili olan popüler bir çekirdek türüdür. RBF çekirdek, 
iki veri noktası arasındaki Öklid uzaklığına dayalı bir benzerlik ölçüsü sağlar ve iki nokta birbirine ne 
kadar yakınsa, çekirdek değeri o kadar büyük olur, böylece verilerin yerel yapısı daha iyi yakalanır. 
Matematiksel olarak 𝑲(𝒙,𝒚) = 𝒆𝒙𝒑(−𝜸 ||𝒙 − 𝒚||𝟐) formülü ile ifade edilir, burada K(x, y) x ve y veri 
noktaları arasındaki çekirdek değerini, γ çekirdeğin genişliğini kontrol eden bir parametreyi ve ||x-y|| x 
ve y vektörleri arasındaki Öklid uzaklığını temsil eder. γ parametresi çekirdeğin ne kadar geniş bir alana 
etki edeceğini belirler; küçük bir γ değeri çekirdeğin sadece yakın noktalara odaklanmasına neden 
olurken, büyük bir γ değeri daha uzak noktaların da etkileşimini sağlar. RBF çekirdek, doğrusal olmayan 
verileri yüksek boyutlu bir uzaya dönüştürerek doğrusal bir hiper düzlem ile ayırmayı sağlar, verilerin 
yerel yapısını yakalar ve birçok farklı veri türünde iyi sonuçlar verir. RBF çekirdek genellikle iyi 
performans gösterir ve aşırı öğrenme riskini azaltabilir, ancak γ parametresinin doğru seçilmesi 
önemlidir ve büyük veri setlerinde hesaplama maliyeti artabilir. Verilerin doğrusal olmayan bir yapıya 
sahip olduğu durumlarda, verilerin yerel yapısının önemli olduğu durumlarda ve diğer çekirdek 
fonksiyonlarının iyi sonuç vermemesi durumunda RBF çekirdek tercih edilir. Özetle, RBF çekirdek 
SVM'lerde doğrusal olmayan verileri sınıflandırmak için güçlü ve popüler bir araçtır, verilerin yerel 
yapısını yakalayarak daha iyi bir model oluşturulmasını sağlar ve doğru parametre seçimi, modelin 
başarısı için kritik öneme sahiptir.

<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/svm_kernels%20intro%20(1).png" width="auto" height="auto">

#### 1.2.8 Algoritmanın Avantajları ve Sınırlamaları
Destek Vektör Makineleri (SVM) algoritması, küçük veri setlerinde iyi performans göstermesi, yüksek boyutlu verilerle çalışma kabiliyeti, az sayıda ayarlanabilir parametre gerektirmesi ve genelleme 
yeteneğinin yüksek olması gibi avantajlara sahiptir; bu avantajlar sayesinde az sayıda veri noktasıyla 
bile iyi genelleme yeteneği gösterir, çekirdek hilesi ile yüksek boyutlu verileri düşük boyutlu bir uzaya 
dönüştürebilir, hiper parametre sayısının az olması modelin eğitimini kolaylaştırır ve maksimum marj 
ilkesi sayesinde aşırı öğrenme riskini azaltır; fakat bunun yanı sıra, SVM'nin büyük veri setlerinde yavaş 
çalışması, hiper parametre seçimi, doğrusal olmayan problemlerde karmaşıklık ve aykırı değerlere 
duyarlılık gibi sınırlamaları da mevcuttur; özellikle veri seti büyüdükçe eğitim süresinin artması, 
çekirdek fonksiyonu ve diğer parametrelerin doğru seçiminin model başarısındaki önemi, çok karmaşık 
doğrusal olmayan problemlerde performansın düşmesi ve aykırı değerlerin hiper düzlemin konumunu 
etkileyerek modelin performansını olumsuz etkilemesi gibi; sonuç olarak, Destek Vektör Makineleri 
(SVM) güçlü bir sınıflandırma algoritması olup, birçok farklı alanda başarılı bir şekilde kullanılmakta 
olup, algoritma seçimi yaparken verinin özellikleri, problemin karmaşıklığı ve istenen performans gibi 
faktörlerin dikkate alınması gerektiği unutulmamalıdır. 

### 1.3 Algoritmanın Uygulama Alanları

#### 1.3.1 Algoritmanın Açıklanması
Aşağıdaki örnek kod, Breast Cancer (meme kanseri) veri seti üzerinde SVM algoritmasını gelişmiş parametrelerle kullanarak en iyi sonucu elde etmeyi amaçlar. Bu veri seti, tıpta sıkça kullanılan bir örnek olduğu için SVM’nin en yaygın kullanım alanlarından birine (sağlık, hastalık teşhisi) ilişkin bir senaryoyu temsil eder.

#### 1.3.2 Algoritma
```python
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# ====================== 1. VERİ HAZIRLAMA ======================
data = load_breast_cancer()
X, y = data.data, data.target

# Eğitim / test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# ====================== 2. PIPELINE OLUŞTURMA ======================
# StandardScaler + SVC
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=42))
])

# GridSearch için hiperparametre aralığı
param_grid = {
    "svc__kernel": ["linear", "rbf", "poly"],
    "svc__C": [0.01, 0.1, 1, 10, 100],
    "svc__gamma": ["scale", 0.1, 1, 10],
    "svc__degree": [2, 3]  # 'poly' kernel için dereceler
}

# ====================== 3. MODEL EĞİTİMİ (GridSearchCV) ======================
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("En iyi parametreler:", grid_search.best_params_)
print(f"En iyi CV skoru: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

# ====================== 4. TAHMİN VE DEĞERLENDİRME ======================
y_pred = best_model.predict(X_test)
accuracy = best_model.score(X_test, y_test)

print("\n-- Test Seti Sonuçları --")
print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["Kanser Yok", "Kanser Var"]))

# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred)
print("Karışıklık Matrisi:\n", cm)

# Tahmin olasılıkları (ROC ve PR eğrileri için)
y_score = best_model.predict_proba(X_test)[:, 1]

# ROC Eğrisi
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Precision-Recall Eğrisi
precision, recall, _ = precision_recall_curve(y_test, y_score)

# ====================== 5. GELİŞMİŞ PLOTLY GÖRSELLEŞTİRME ======================
# 3 alt grafik (1 satır, 3 sütun): 
#   1) Karışıklık Matrisi, 2) ROC Eğrisi, 3) Precision-Recall Eğrisi
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=["Karışıklık Matrisi", "ROC Eğrisi", "Precision-Recall Eğrisi"],
    horizontal_spacing=0.08
)

# -----------------------------------------------------------
# (a) Karışıklık Matrisi (Heatmap)
# -----------------------------------------------------------
# Confusion Matrix'i DataFrame'e dönüştürerek annot için metin hazırlama
cm_df = pd.DataFrame(cm, index=["Gerçek: Yok", "Gerçek: Var"], columns=["Tahmin: Yok", "Tahmin: Var"])
annot_text = cm_df.values.astype(str)

heatmap = go.Heatmap(
    z=cm_df.values,
    x=cm_df.columns,
    y=cm_df.index,
    colorscale="Blues",
    showscale=True,
    text=annot_text,
    texttemplate="%{text}",
    textfont={"size": 14},
    hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Adet: %{z}<extra></extra>"
)

fig.add_trace(heatmap, row=1, col=1)

# -----------------------------------------------------------
# (b) ROC Eğrisi
# -----------------------------------------------------------
roc_curve_trace = go.Scatter(
    x=fpr,
    y=tpr,
    mode="lines",
    line=dict(color="firebrick", width=3),
    name=f"ROC AUC = {roc_auc:.2f}",
    hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>"
)

# Rastgele sınıflandırma çizgisi
roc_line = go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode="lines",
    line=dict(color="gray", dash="dash"),
    showlegend=False
)

fig.add_trace(roc_curve_trace, row=1, col=2)
fig.add_trace(roc_line, row=1, col=2)

# -----------------------------------------------------------
# (c) Precision-Recall Eğrisi
# -----------------------------------------------------------
pr_curve_trace = go.Scatter(
    x=recall,
    y=precision,
    mode="lines",
    line=dict(color="green", width=3),
    hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>",
    name="Precision-Recall"
)

fig.add_trace(pr_curve_trace, row=1, col=3)

# ====================== 6. DÜZENLEMELER ======================
# Genel başlık ve layout ayarları
fig.update_layout(
    title_text="<b>SVM Performans Analizi (Breast Cancer)</b>",
    title_x=0.5,  # Ortaya hizalama
    font=dict(size=14),
    plot_bgcolor="white"
)

# X/Y eksen etiketleri
fig.update_xaxes(title_text="False Positive Rate", row=1, col=2)
fig.update_yaxes(title_text="True Positive Rate", row=1, col=2)

fig.update_xaxes(title_text="Recall", row=1, col=3)
fig.update_yaxes(title_text="Precision", row=1, col=3)

fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

# Görseli etkileşimli göstermek için
fig.show()

```
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/3Ekran%20Al%C4%B1nt%C4%B1s%C4%B1.PNG" width="auto" height="auto">
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/4Ekran%20Al%C4%B1nt%C4%B1s%C4%B1.PNG" width="auto" height="auto">
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/1Ekran%20Al%C4%B1nt%C4%B1s%C4%B1.PNG" width="auto" height="auto">
<img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/2Ekran%20Al%C4%B1nt%C4%B1s%C4%B1.PNG" width="auto" height="auto">

### Kodun Açıklaması
1. **Veri Seti (load_breast_cancer):**  
   - `load_breast_cancer()` fonksiyonu, meme kanseri tanı verilerini içeren bir veri seti döndürür.  
   - `X`, gözlem değerlerini (hücre ölçümleri vb.) içerirken, `y` etiket (kanser olup olmadığı) bilgilerini tutar.

2. **Eğitim ve Test Ayırma (train_test_split):**  
   - Veri seti, modelin başarısını doğru ölçebilmek için eğitim ve test olarak ikiye ayrılır (%70 eğitim, %30 test).  
   - `stratify=y` ile sınıf dengesini (örneklerin sınıflar arasında eşit dengede olmasını) koruyoruz.

3. **Pipeline ve StandardScaler:**  
   - `StandardScaler`, özelliklerin ortalamasını 0, varyansını 1 olacak şekilde dönüştürerek veriyi ölçeklendirir.  
   - Ardından `SVC(probability=True)` ile SVM modelini kurarız. `probability=True`, sınıflandırma sonucunda olasılık değerlerini de hesaplamayı sağlar (örneğin ROC eğrisi çizmek için gerekli olabilir).

4. **Hiperparametre Arama (Param Grid):**  
   - **`kernel`:** SVM’de çekirdek fonksiyonunu belirler (`linear`, `rbf`, `poly`).  
   - **`C`:** Düzenleme (regularization) parametresi, model karmaşıklığını kontrol eder.  
   - **`gamma`:** `RBF` veya `poly` kernel kullanıldığında, veri noktalarının ne kadar yakın olması gerektiğini belirleyen parametredir.  
   - **`degree`:** `poly` kernel için polinom derecesidir.

5. **GridSearchCV ile Çapraz Doğrulama:**  
   - Oluşturulan her parametre kombinasyonu 5 katlı çapraz doğrulama (cv=5) ile test edilir.  
   - `scoring="accuracy"`, en iyi modeli doğruluk oranına göre seçer.  
   - `n_jobs=-1`, kullanılabilir tüm işlemci çekirdeklerini kullanır (büyük veri setlerinde süreyi kısaltmak için faydalı).

6. **En İyi Modelin Eğitilmesi ve Değerlendirme:**  
   - `grid_search.best_estimator_` ile en iyi bulunan model seçilir.  
   - Test veri seti üzerinde tahmin (`predict`) işlemi yapılarak modelin performansı ölçülür.  
   - `classification_report`, sınıf bazında kesinlik (precision), bulma (recall) ve F1 skorunu gösterir.  
   - `confusion_matrix`, tahmin edilen etiketler ile gerçek etiketler arasındaki eşleşmeyi gösterir.

  <img src="https://github.com/ncrim7/support-vector-machines/blob/main/img/Ekran%20Al%C4%B1nt%C4%B1s%C4%B1.PNG" width="auto" height="auto">


#### 1.3.3 Kullanım Alanları
Bu kod ile meme kanseri verilerinde SVM algoritmasının en uygun parametrelerini arayarak daha isabetli bir teşhis modeli elde edebilirsiniz. Aynı yöntemi, e-posta spam tespiti, finansal veri tahminleri veya yüz tanıma gibi farklı veri setlerine uyarlayarak SVM’nin yaygın kullanım alanlarındaki performansını da inceleyebilirsiniz.
### 1.4 Performans Analizi

#### 1.4.1 Uzay ve Zaman Karmaşıklığı
Zaman Karmaşıklığı 
SVM'nin zaman karmaşıklığı, kullanılan çekirdek fonksiyonuna (kernel) ve veri boyutuna bağlı olarak 
değişir. Kodunuzda kullanılan lineer çekirdek (kernel='linear'), SVM'nin daha basit ve hızlı 
çalışmasını sağlar. Zaman karmaşıklığı şu şekilde hesaplanabilir: 
Eğitim Aşaması (fit): 
Zaman karmaşıklığı, veri sayısına (n) ve özellik sayısına (d) bağlıdır. 
Lineer çekirdek için karmaşıklık: 𝑶(𝒏𝟐 ⋅ 𝒅)𝑶(𝒏^𝟐 \𝒄𝒅𝒐𝒕 𝒅)𝑶(𝒏𝟐 ⋅ 𝒅) 
Eğer veri seti çok büyükse (örneğin, milyonlarca veri noktası), eğitim süresi önemli ölçüde artar. 
Tahmin Aşaması (predict): 
Lineer çekirdek için her bir veri noktası tahmin edilirken zaman karmaşıklığı:  
�
�(𝒏⋅𝒅) 
Genel Durum: 
Ortalama Durum: 𝑶(𝒏.𝒍𝒐𝒈𝒏) 
En Kötü Durum: 𝑶(𝒏𝟐) Özellikle, veri setinin doğrusal olarak ayrılabilir olmadığı durumlarda eğitim 
süresi artabilir. 
Uzay Karmaşıklığı 
SVM, eğitim sırasında verileri destek vektörleriyle temsil eder. Tüm veri setini değil, yalnızca destek 
vektörlerini saklar. 
Uzay karmaşıklığı, veri boyutuna (n) ve özellik sayısına (d) bağlıdır: O(𝒏 ⋅ 𝒅) 

#### 1.4.2 Optimizasyon Seçenekleri
Kernel linear yerine parabolik bölüm yapan rbf ve polynomial olabilir. 
GridSearchCV ile fazla öğrenmeyi önleyebiliriz. 
Test size değişebilir. Bizim örnek için 0.2. 
Veriler eğitime girmeden fit edilebilir. 
### 1.5 Çalışma Soruları ve Egzersizler
##### Problem 1: 
- Müşterilerin yaşı, bakiyesi, son harcamaları kredi skoru gibi özelliklerine bakarak farklı etiketlerdeki 
müşteri sınıflarına ayırmak. 
##### Çözüm1: 
- Veri setindeki gerekli yerleri numerik yapıp  fit edilip linear kernel ile eğitme. 
##### Problem 2: 
- Hastanın yaşı, cinsiyeti, boyu, kilosu gibi özelliklerine bakarak hasta riski olup olmadığını hesaplama. 
##### Çözüm2: 
- Veri setindeki gerekli yerleri numerik yapıp  fit edilip RBF kernel ile eğitme. 
##### Problem 3: 
- El yazısından rakamları tanıma. 
##### Çözüm3: 
- Veri setindeki resimlere ve verilen etiketlerine bakılır resimler arasındaki benzerlik ile eğitilir fit  edilip 
RBF kernel ile eğitme. 

### 1.6 Algoritma Özeti
Destek Vektör Makineleri (SVM), 1960'larda Vladimir Vapnik ve Alexey Chervonenkis tarafından 
geliştirilmeye başlanmış ve 1990'larda Bernhard Boser, Isabelle Guyon ve Vladimir Vapnik tarafından 
tanıtılan çekirdek hilesiyle (kernel trick) gerçek potansiyeline ulaşmış bir makine öğrenmesi 
algoritmasıdır; doğrusal ve doğrusal olmayan verilerin sınıflandırılması, regresyon, anomali tespiti ve 
kümeleme gibi çeşitli görevlerde kullanılır. SVM, küçük veri setlerinde iyi performans göstermesi, 
yüksek boyutlu verilerle çalışabilmesi, az sayıda ayarlanabilir parametre gerektirmesi ve genelleme 
yeteneğinin yüksek olması gibi avantajlara sahiptir; bu avantajlar sayesinde az sayıda veri noktasıyla 
bile iyi genelleme yeteneği gösterir, çekirdek hilesi ile yüksek boyutlu verileri düşük boyutlu bir uzaya 
dönüştürerek karmaşık problemleri çözebilir, ancak büyük veri setlerinde yavaş çalışma, hiperparametre 
seçimi, doğrusal olmayan problemlerde karmaşıklık ve aykırı değerlere duyarlılık gibi sınırlamaları da 
vardır. SVM algoritması, veri noktalarını en büyük marjla ayıran bir hiper düzlem bulmayı hedefleyen 
bir optimizasyon problemi çözer; bu problem, kısıtlı kuadratik programlama olarak bilinir ve özel 
algoritmalar ve yazılımlar gerektirir. Çekirdek hilesi, verileri açıkça yüksek boyutlu bir uzaya 
dönüştürmeden, çekirdek fonksiyonları aracılığıyla bu dönüşümü ima eder ve yüksek boyutlu 
uzaylardaki hesaplamalar yerine çekirdek matrisi üzerinde işlemler yaparak hesaplama karmaşıklığını 
azaltır; farklı çekirdek fonksiyonları, doğrusal olmayan veri kümelerini başarılı bir şekilde 
sınıflandırmak için kullanılır. Polinom çekirdek, iki veri noktası arasındaki ilişkiyi bir polinom 
fonksiyonu ile ifade ederken, Radyal Temel Fonksiyon (RBF) çekirdek, iki veri noktası arasındaki Öklid 
uzaklığına dayalı bir benzerlik ölçüsü sağlar ve bu çekirdekler, doğrusal olmayan verilerin daha yüksek 
boyutlu bir uzaya dönüştürülerek daha iyi sınıflandırılmasını sağlar. SVM'nin kullanım alanları arasında 
görüntü tanıma, el yazısı tanıma, yüz tanıma, nesne tanıma, spam e-posta tespiti, duygu analizi, metin 
sınıflandırma, biyomedikal uygulamalar, finansal analizler ve ses tanıma gibi çeşitli uygulamalar yer 
alır. Performans analizi açısından SVM'nin zaman ve uzay karmaşıklığı, kullanılan çekirdek 
fonksiyonuna ve veri boyutuna bağlıdır; örneğin, lineer çekirdek kullanıldığında zaman karmaşıklığı 
�
�(𝒏𝟐 ∗ 𝒅) ve uzay karmaşıklığı 𝑶(𝒏 ∗ 𝒅) olarak hesaplanabilir. Optimizasyon seçenekleri arasında 
farklı çekirdek fonksiyonlarının (örneğin, polinomsal veya RBF çekirdek) kullanılması, GridSearchCV 
ile aşırı öğrenmenin önlenmesi ve veri setinin eğitim verisi ve test verisi olarak bölünmesi (örneğin, 
%20 test verisi) bulunur. Ayrıca, SVM algoritması, kod örneği olarak verilen iris veri seti üzerinde de 
gösterildiği gibi, eğitim ve test verilerini ayırarak, lineer çekirdek ile eğitim yaparak ve doğruluk oranını 
hesaplayarak kullanılabilir. Destek Vektör Makineleri, doğru parametrelerin seçilmesi ve modelin aşırı 
öğrenme riskinin azaltılması ile çeşitli alanlarda başarılı bir şekilde uygulanabilir.
