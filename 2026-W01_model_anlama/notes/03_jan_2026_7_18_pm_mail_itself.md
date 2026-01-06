Hasan, sana daha önce bahsettiğim gibi bu kez Muller 2024 verilerine uygun şekilde tüm kimyasal türleri her bir kayaç ve sıcaklık için küçük RMSE hata paylarıyla tutturmayı başardım. Ekteki zipli dosyada tüm tür kayaç ve herbiri içinde 3 farklı sıcaklık fitleri ve sonuçlarını görebilirsin (çıktı figürler, scriptler, çıktı data dosyaları vs). Artık bu kodları ve verileri kullanabilirsin. En etkin fiziksel yaklaşım ve matematiksel çözüm bu.

 

Tüm kayaç türleri ve sıcaklıkları tek bir algoritma üstünde hesaplamaya çalıştım. Sadece belirli parameter değerleri değiştirerek fitleri elde edebiliyoruz. Scriptlerde hepsinde rate değerleri .dat dosyalarına yazdırıyorum.

 

Bu fitleri yaparken daha önce sana söylediğim gibi aslında sistemi çift faz (two-phase gas-liquid) olarak düşünmem gerekiyordu çünkü kimyasal türler (H2S, H2 vs) bunlar düşük basınç altında hem gaz hem sıvı halde Mullerin sisteminde bulunuyordu. Öncelikle bu fiziği ekledim ve kimyadaki Henrys solubility yasasını dikkate aldım. Aynı zamanda reaksiyon kinetiklerinide kayaçlardaki mineral türlerini dikkate alarak düzenledim. Mesela özellikle gypsum (jips) kayacı kimyasal formülü CaSO4·2H2O yani çözündüğünde ortama çok fazla SO4 yani sülfat salıyordu bu da hidrojen sulfate reduction’I daha fazla tetikliyordu. Diğer yandan demiroksitle temasında çok erken H2S gaz salınımına neden oluyordu. Bunları her bir kayaç ve sıcaklıkta girdiğim parametreleri adjust etmem gerekiyordu. Şunuda belirtmem gerekirki sofistike jeokimyasal hesaplama programları size bahsettiğim phreeqc gibi bu derece bir fiziksel yaklaşımı çiftfazlı çözebilmesi bu kadar iyi fit edemeyebilirdi ve matematiksel background hesaplamalarına bu derece müdahele edemeyebilirdim. Elimizde direk fiziksel çözümün olması iyi oldu.

 

Dediğim gibi tüm scriptler eşdeğer sadece en önemli sensitivity parametreleri henry’s solubility contant değerleri (H2, CO2, ve H2S için), initial SO4, pH değerleri. Bunları scriptlerin içinde comment olarak yazdım. O yüzden algoritmayı bir ama farklı kayaçlar için sadece bu parametreleri copy paste etmen yeterli. Bu parametreler özellikle fiti çok hassas şekilde etkileyenler,  Hcp_H2_base, Hcp_CO2_base, Hcp_H2S_base, S_tot0, env.pKa_H2S, env.SO4_sat_gyp.

 

Bu scriptleri CPU kullanarak çözdüm, bu biraz zaman alıyor açıkcası. Matlabda ben GPU’da kullanabiliyorum ama kodu değiştirmek istemedim. Eğer seninde gpu kullanımın varsa bunu modifiye edersen phyton için çok daha hızlı bir çözüm olur.

 

Şimdi benim için, düşük basınçdaki Muller 2024 verileri aslında bize güzel bir başlangıç oldu ve neural training içinde bence basınç aralıkları ve farklı kayaç/mneral türleri için geniş bir yelpazede sonuç verebilir diye düşünüyorum. Buradan yola çıkarak şimdi yüksek basınç altında gerçekleştirilmiş deneylere odaklanacağım. Bunlarla ilgili çalışmalarıda peyderpey size gönderirim. Bunun dışında belki bende neural network kısımlarında bişeyler elde edebilirsem bunlarıda sizinle paylaşırım. Muller 2024 ile ilgili yapılabilecek tüm şey bu kadar 