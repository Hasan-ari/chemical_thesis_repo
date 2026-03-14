Merhaba Hasan,

Umarım iyisiniz.

Size bahsettiğim gibi Mura 2024 makalesini baz alarak phreeqc kodunu yaptım ekte görebilirsiniz (karşılaştırma için okuması gereken dosya il eve phreeqc dosyasını aynı zamanda text dosyası olarakda kaydettim bakmak isterseniz). Yaklaşık 2 hafta uğraştım, bunun bir haftasını aslında CO-pilot/ChatGPT ile phreeqc yazdırabilirmiyiz diye uğraştım ancak tam bir fiyasko. Kodu üretiyor ancak hiçbir şekilde tutarlılığı yok üstüne yanlış yönlendiriyor. Birde Mura 2024’de bilgilerin eksik yada belirsiz olmasından bende sonuca ulaşmakta zorlandım en sonunda kendi bildiğim gibi yaptım. Phreeqc’yi kullanmak isterseniz aslında basit, notepad++ üstünde çalışıyor ve USGS sayfasından Phreeqc extension indirip kullanıyorsunuz, temelde yine kod yazılıyor ama phreeqc’nin anlayabileceği blokları yazmanız gerekiyor.

 

Hatırlatmam gerkirse aslında Matlabda yaptığım şey redox reaksiyon tepkimelerini deneysel verilere fit ederek sadece aslında monod-kinetic verilerini elde ettim (methanogenesis, acetogenesis vs reaksiyonlarında gereken reaksiyon rateleri half saturation vs.) phreeqc’de kullanmak üzere soransında mineral kinetiklerinide dikkate alarak deneyde gerçekleşenleri re-generate etmeye çalışıyorum, yani phreeqcde bişeyleri otomatik fit etmiyorum simulasyon yaparak aynı trendleri ve değerleri yakalamaya çalışıyorum.

 

Şu anda hidrojeni, formate, ve calcium’u deney sonucuyla yakın tutturabiliyorum ancak biraz optimizasyon gerekiyor ve bu aşamada bizim ekiple (Berkay, Zelal ve Mehmet) halledeceğim. Onlarla beraber Matlab ya da Phyton’da (onlarda phyton kullanıyor), içine Phreeqc’yi entegre edip parameter sensitivitesi yapmamız gerekiyor (reaksiyona giren mineral yüzey alanı özellikle burada calcite ve barite çok etkin).

 

Mura 2024 de bahsedilen protocol ve bazı sonuçlar tam olarak phreeqc hesaplarıyla uyuşmuyor. Özellikle 60 bar basıncı büyük ihtimal azot basarak sağlıyorlar yada Tolune basarak ama tam anlatmamışlar. Headspace ne kadar volume belli değil, H2 injection 0.105 mmol diyor ama partial pressure verilmemiş bunları ben kendim ayarlayarak 0.105 mmol değeri buldum. Ayrıca makalede 99 CH4+CO2 basıldığından bahsediyor ama figure 1 de CO2 değeri 40 mmol görünüyor, bu değer aslında bence CH4+CO2 ortak değeri sanırım çünkü hidrojen basıldığında çok az miktarda olan CO2 zaten hidrojenle direk tepkimeye girip metan oluşturuyor (tam tersi CO2 çok basılsada hepsi hidrojenle metana dönüyor) o yüzden burada Mura’nın tanımlamasında yada sonuçlarında bir hata var. phreeqc’deki problemler/limitasyonlarıda iyice denedim, sonrasında sizlerede anlatabilirim. Bu kod içinde hesaplama Mura’nın 9’uncu gününde hidrojen enjeksiyonuyla başlıyor.

 

Önümüzdeki 2 hafta bu kod üzerinden Mura 2024 verilerini baz alarak sensitivite yapacağız, bu işi halledelim sonrasında buradan çıkan bütün verileri ve daha fazlasını sonrasında deneyerek (farklı mineral yüzdeleri, initial conditionlar vs) size neural network training için vereceğiz. Yine detaylı şekilde bi toplantı yaptığımızda anlatabilirim tartışabiliriz. Phreeqc’yi phtonla birleştirdiğimizde otomatik olarak çok fazla sentetik veri üretebiliriz.

 

Selamlar