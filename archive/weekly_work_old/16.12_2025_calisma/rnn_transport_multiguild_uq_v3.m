function rnn_transport_multiguild_uq_v3
% - Mechanistic ODEs for microbial guilds (methanogenesis, sulfate reducers, acetogenesis)
% - Parameter fitting to lab data (H₂, CH₄, H₂S, SO₄, CO₂)
% - LSTM neural network to emulate microbial reaction dynamics in a 1D transport column
% - Reactive transport simulation with advection, dispersion, and uncertainty quantification

    %% Load experimental data
    raw = readmatrix('Muller_2024_H2_Sandstone_at_25C.txt');
    t_exp = raw(:,1); % time in days
    % concentrations of chemical species (H2, CO2, CH4, H2S) at columns from 2 to 5 are in µmol, at column 6 (SO4) is in already in mmol  
    data_exp = [raw(:,2:5)*1e-3, raw(:,7)]; % Convert µmol to mmol
    disp(['data exp : ' ...
        'H2   CO2  CH4  H2S SO4(mM)' ...
        '        '])% 
    disp([data_exp])

    %% Initial condition: [H2, CO2, CH4, H2S, SO4,]
    % Added new 5 more FeS (Precipitated iron sulfide) , X_meth (Methanogen biomass), X_sulf (Sulfate reducer biomass), X_aceto (Acetogen biomass), Acetate (Acetic acid)]
    x0 = [data_exp(1,:)'; 0.01; 0.01; 0.01; 0; 0]; % 10 elements 
    disp('x0 : ')
    disp([x0])

    %%
    %Sıra,Değişken,Değer,Anlamı x0
    % 1-5,"H2, CO2...",Tablodan,Ölçülen kimyasallar.(Gerçek lab ölçümleri)
    % 6,FeS,0.01,Demir Sülfür çökeltisi. Ortamda başlangıçta çok az var diyoruz.
    % 7,X_meth,0.01,"Metanojen Bakterisi. Hiç yok diyemeyiz (yoksa üreyemezler), ""tohum"" kadar var."
    % 8,X_sulf,0.01,"Sülfat İndirgeyici Bakteri. Yine ""tohum"" miktarı."
    % 9,X_aceto,0.01,"Asetojen Bakterisi. (Kodda 0 yazılmış olabilir ama genelde biyokütle 0 olmaz, buradaki 0 o anlama gelebilir ya da modelde r hız denklemi biyokütleye bağlı değilse 0 dan başlayıp artabilir). Düzeltme: Kodda burası 0 olarak girilmiş."
    % 10,Acetate,0,Asetat. Başlangıçta ortamda hiç asetat yok.


    %% Fit mechanistic parameters and simulate trajectory
    %Giriş parametrelerinin açıklaması:
    %t_exp 
    disp('t_exp :')
    disp(t_exp)
%%
    p_fit = fit_mechanistic_params(t_exp, data_exp, x0);%Parametreleri eğittik gerçek simülasyon ile.

    disp('p_fit :')
    disp(p_fit)



%%


    %{
    Bu blok, projenin Veri Çoğaltma (Data Augmentation) fabrikasıdır.
Satır satır ne olduğuna bakalım:
1. Yeni Bir Zaman Çizelgesi Oluşturmak
t = linspace(0, t_exp(end), 2000);
Durum: Elimizdeki gerçek veri (t_exp) sadece ~10-15 satırdı (0. gün, 1.1. gün...). Bu, bir Yapay Zeka (LSTM) eğitmek için çok azdır.
İşlem: linspace komutu ile 0. günden son güne kadar olan süreyi 2000 eşit parçaya bölüyoruz.
Amaç: "Bize 15 karelik bir video değil, 2000 karelik akıcı bir film lazım."   
    %}

    t = linspace(0, t_exp(end), 2000); %Linear space .Row vector of equally spaced values.
    %Sayı doğrusundaki aralıkları eşit olan noktaları düşün.
    %Örnek : linspace(0,1,10) = 0 , 0.1111,0.2222,0.3333,0.4444 , 0.5556,
    %0.6667, 0.7778, 0.8889,1.000 1 ve 10 arasını 10 eşit parçaya bölmüş
    %olduk.




 

    %%
    %Burada en son veri üreten simülasyonumumuzu yaptık 2000 defa kadar. Ve
    %başlangıç değerlerimiz x0
    
    %Burda p_fitin içinde zaten en doğru paramterler var bu
    %[1,1,1.......-10]dan en doğru haline büründü :
    %{
    p_fit :
  Columns 1 through 12

    0.8739    0.0263    0.0043    0.0500    0.0500    0.0500    0.0023    0.0031    0.5208    0.7491    0.0000    0.1262

  Column 13

  -10.0000
    
    %}

    [~, xTrain] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p_fit), t, x0);
    %Final data üretme simülasyonu.
    %Her adımda hızı yani p_fit leri bildiğimiz için biz o oranı tamamen
    %küçük zaman dilimlerine uygulayabiliyoruz. Şimdi anlaşıldıııı


    %{    
 Hedef: 2000 Karelik Film ÇekmekElimizde t = linspace(0, 10, 2000) var.
 Yani süremiz 10 saat ama biz bunu 2000 küçük parçaya bölüyoruz.
Adım Aralığı (dt): $10 / 2000 = \mathbf{0.005}$ saat (Yani her 18 saniyede bir ölçüm alacağız).
Bilgisayar (ode45), 
1. satırdan başlayıp 2000. satıra kadar şu döngüyü kurar:
Kare 1 (Başlangıç - t = 0.000)Durum: H2 = 100, Bakteri = 2.Kayıt: xTrain tablosunun 1
. sütununa bu sayıları yazar.
Kare 2 (t = 0.005) - Hesap Anı 
Bilgisayar sorar: "Bir önceki karede (t=0) durum neydi?
"Cevap: 2 Bakteri vardı.Hesap (Türev): Değişim = -2 * 2 = -4 (Saatte 4 birim H2 azalmalı).
Uygulama (Euler Adımı): Ama sadece 0.005 saat geçti.Azalma = Hız (-4) * Süre (0.005) = -0.02 birim.
Yeni Durum:H2 = $100 - 0.02 = \mathbf{99.98}$Bakteri = 2 (Şimdilik artmadığını varsayalım).
Kayıt: xTrain tablosunun 2. sütununa [99.98; 2] yazar.
Kare 3 (t = 0.010)Bilgisayar yine sorar: "Bir önceki karede (t=0.005) durum neydi?"
Cevap: Hala 2 Bakteri var.Hesap: Hız yine -4.Yeni Durum:H2 = $99.98 - 0.02 = \mathbf{99.96}$
Kayıt: 3. sütuna yazar.... (Aradan zaman geçer, bakteriler çoğalır) 
...Kare 1000 (t = 5.000)
Durum: Diyelim ki Bakteriler çoğaldı ve 50 kişi oldular.
 H2 ise 40'a düştü.Hesap (Yeni Hız): Değişim = -2 * 50 = -100 (Artık çok hızlı yiyorlar!).
Uygulama: Azalma = -100 * 0.005 = -0.5 birim.
Yeni Durum:H2 = $40 - 0.5 = \mathbf{39.5}$Kayıt: 1000. sütuna [39.5; 50] yaza  
    
    %}
    
    %%
    xTrain = xTrain'; % [10 x timeSteps] % Sadece formatlama var. ' işareti ile matrisi yan yatırdık.


%{


Sonuç: xTrain Matrisi Neye Benziyor?İşlem bittiğinde hafızada şöyle dev bir matris (tablo) oluşur:
Özellik / Zamant=0 (Kare 1)t=0.005 (Kare 2)...t=5.0 (Kare 1000)...t=10 (Kare 2000)
H2100.0099.98...39.50...0.01Bakteri2.002.00...50.00...95.00CO2..................
%}

%% Prepare sequences for LSTM training (sequence-to-one)
sequenceLength = 10; % number of past steps to use as input
X = {}; % cell array of sequences
Y = []; % numeric matrix of next-step targets

for i = 1:(size(xTrain,2) - sequenceLength)
    % Each predictor is a [features x sequenceLength] matrix
    X{end+1} = xTrain(:, i:i+sequenceLength-1);

    % Each response is a single [1 x features] row vector
    Y(end+1, :) = xTrain(:, i+sequenceLength)'; 
end

% Define LSTM network
layers = [ ...
    sequenceInputLayer(size(xTrain,1))  % number of features = number of states
    dropoutLayer(0.2)
    lstmLayer(64,'OutputMode','last')   % output only last time step
    dropoutLayer(0.2)
    fullyConnectedLayer(size(xTrain,1)) % predict all features at next step
    regressionLayer];

% Training options
options = trainingOptions('adam', 'MaxEpochs', 300, 'MiniBatchSize', 64, 'InitialLearnRate', 1e-3, 'Shuffle','every-epoch', ...
    'Verbose',false);

% Train sequence-to-one model
net = trainNetwork(X, Y, layers, options);

% Save trained network
save('trained_LSTM_multiguild.mat','net');
   
% --------------------------1D REDOX REACTIVE TRANSPORT ZONATION------------------------------
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% Spatial and temporal discretization--------------------------------------
 %% Transport simulation with ensemble sampling
L=75;                                         % length of column [m]
N=75;                                         % Number of cells
cell_w=1;                                     % Cell width [m]
S_time=t_exp(end);                                   % Simulation time [d]
% Spatial Discretization --------------------------------------------------
 x=cell_w:cell_w:L;
 nx = length(x);
% Flow and transfer parameters---------------------------------------------
% Confined aquifer
n=0.3;                                       % porosity [-]
q=1;                                         % Darcys velocity [m/d]
v=q/n;                                       % seepage velocity [m/d]
D=0.3;                                       % Dispersion coefficient [m^2/d]
alpha=D/v;                                   % dispersivity [m]

% Travel time in each cell -------------------------------------------------
dt=cell_w/v;                                       % [d]

% Matrix of Concentrations 
% Rows related to length coordinates
% Columns related to components
    cmob = zeros(nx, 6); % H2, CO2, CH4, H2S, SO4, Acetate, % mobile species
    cimob = zeros(nx, 4); % FeS, X_meth, X_sulf, X_aceto  % imobile species
    cmob(:,1:2) = 1e-4; cmob(:,5) = 5e-5;
    cimob(:,1:4) = 0.01;

    % Breakthrough curve (BTC) matrix for 10 species
    BTC_mean = zeros(0,10); BTC_std = zeros(0,10);

    historyLength = sequenceLength;
    historyBuffer = repmat(x0,1,historyLength);
%% Loop over all timepoints------------------------------------------------
    for time = 0:1:S_time

    %% -------------------------ADVECTION----------------------------------------
% Advection a Courant-number 1 implies that the concentrations are
% moved by exactly one box. The values in the last box are moved out.
% The first box receives the inflow concentration.
% Shifting mobile concentrations to the beginning of matrix
        cmob(2:end,:) = cmob(1:end-1,:);
        cmob(1,:) = [1e-4, 1e-4, 0, 0, 5e-5, 0];
%% ------------------------DISPERSION----------------------------------------

% Calculation of dispersive fluxes at the interior interfaces-------------

        Jd = (cmob(1:end-1,:) - cmob(2:end,:)) / cell_w * D;
% Add a dispersive flux of zero at the inflow boundary and assume that
% the dispersive flux at the outflow is identical to that at the last
% internal interface
        Jd = [zeros(1,6); Jd; Jd(end,:)];
%  Concentration change due to divergence of dispersive flux--------------        
        cmob = cmob + dt/cell_w * (Jd(1:end-1,:) - Jd(2:end,:));
%% ---------------------------REACTION-------------------------------------
% compute rate of change due to reaction-----------------------------------
% A new concentration zero matrix for using after ODE
        cmat_ensemble = zeros(nx,10,20); % 20 samples
        % For the ODE solver it creates matrix for each time duration  
        for it = 1:nx
            currentState = [cmob(it,:), cimob(it,:)]';
            historyBuffer = [historyBuffer(:,2:end), currentState];

            for s = 1:20
                y_pred = predict(net, historyBuffer, 'ExecutionEnvironment','cpu');
                cmat_ensemble(it,:,s) = y_pred;
            end
        end

        cmat_mean = mean(cmat_ensemble,3);
        cmat_std = std(cmat_ensemble,0,3);

        cmob = cmat_mean(:,1:6);
        cimob = cmat_mean(:,7:10);

        BTC_mean = [BTC_mean; cmat_mean(25,:)]; % Collect data at 25 m
        BTC_std = [BTC_std; cmat_std(25,:)];  % Collect data at 25 m
    end

    %% Plot BTC with uncertainty
    tvec = 0:1:S_time;
    species = {'H2','CO2','CH4','H2S','SO4','Acetate','FeS','X_meth','X_sulf','X_aceto'};

    figure;
    for i = 1:10
        subplot(5,2,i)
        % shadedErrorBar(tvec, BTC_mean(:,i), BTC_std(:,i), 'lineProps', '-b');
        % title(['BTC: ', species{i}]); xlabel('Time [d]'); ylabel('mmol/L');
        % Compute upper and lower bounds
upper = BTC_mean(:,i) + BTC_std(:,i);
lower = BTC_mean(:,i) - BTC_std(:,i);

% Fill the shaded area
fill([tvec fliplr(tvec)], [upper' fliplr(lower')], [0.8 0.8 1], ...
     'EdgeColor','none', 'FaceAlpha',0.3); 
hold on;

% Plot the mean line
plot(tvec, BTC_mean(:,i), '-b', 'LineWidth', 1.5);
% if i<=5
% plot(t_exp(:,1) ,data_exp(:,i), 'ro', 'LineWidth', 1.5)
% else
% end
title(['BTC: ', species{i}]); xlabel('Time [d]'); ylabel('mmol/L');
    end
end

%t_exp :  gerçek verideki zaman verileri 0 , 1,1 , 5 ,6 ....19.0
%data_exp :  gerçek lab verileri H2 , CO2 , CH4 ,H2S SO4(mM)
%x0 :  ise data_exp (H2 , CO2 , CH4 ,H2S SO4(mM) + 5 tane ilk değer
%0.0100,0.0100,0.0100, 0 , 0 Bunlar ise  
    % 6,FeS,0.01,Demir Sülfür çökeltisi. Ortamda başlangıçta çok az var diyoruz.
    % 7,X_meth,0.01,"Metanojen Bakterisi. Hiç yok diyemeyiz (yoksa üreyemezler), ""tohum"" kadar var."
    % 8,X_sulf,0.01,"Sülfat İndirgeyici Bakteri. Yine ""tohum"" miktarı."
    % 9,X_aceto,0.01,"Asetojen Bakterisi. (Kodda 0 yazılmış olabilir ama genelde biyokütle 0 olmaz, buradaki 0 o anlama gelebilir ya da modelde r hız denklemi biyokütleye bağlı değilse 0 dan başlayıp artabilir). Düzeltme: Kodda burası 0 olarak girilmiş."
    % 10,Acetate,0,Asetat. Başlangıçta ortamda hiç asetat yok.
function p_fit = fit_mechanistic_params(t_exp, data_exp, x0)
    % Define parameter bounds and initial guess
    p0 = [1, 1, 1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, -10];

     %{
        1. p0 (İlk Tahmin / Initial Guess)
p0 = [1, 1, 1, ... -10];

Bu vektördeki 13 sayı, 13 farklı fiziksel kuralın başlangıç değeridir.

Bu 13 sayı şunları temsil eder (kodun ilerleyen kısımlarındaki trueODEfunc fonksiyonundan biliyoruz):

İlk 3 sayı (1,1,1): Reaksiyon Hızları (Bakteriler ne kadar hızlı yiyor?).

Sonraki 3 sayı (0.05...): Verimler (Yedikleri yemeğin ne kadarı vücutlarına, ne kadarı enerjiye gidiyor?).

Diğerleri: Engelleme katsayıları, Çökme hızları vb.

Son sayı (-10): Termodinamik Enerji eşiği.

"Araştırmaya buradan başla, hızları 1 civarında, verimleri 0.05 civarında düşün."     
     %}
    % 2. Sınırlar (Lower & Upper Bounds)
    lb = [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0, 0, 0, -50];
    ub = [10, 10, 10, 0.5, 0.5, 0.5, 10, 10, 10, 1, 1, 1, 0];

    %{
lb (En az): Hiçbir hız 0.001'den küçük olamaz. Hiçbir verim 0.01'den az olamaz.
ub (En çok): Hiçbir hız 10'dan büyük olamaz (Bakteriler ışık hızında yemek yiyemez).   
    %}

    options = optimoptions('lsqnonlin','Display','iter','MaxFunctionEvaluations',5000);
        %{
        lsqnonlin: Kullanacağımız algoritmanın adı (Detayına aşağıda geleceğiz).
        Display, iter: "Her adımda ne bulduğunu bana ekrana yaz, gizli çalışma."
        MaxFunctionEvaluations, 5000: "En fazla 5000 kere deneme yap. Bulamazsan pes et, sonsuz döngüye girme."
        %}




    p_fit = lsqnonlin(@(p) residuals_multiguild(p, t_exp, data_exp, x0), p0, lb, ub, options);
end


function res = residuals_multiguild(p, t_exp, data_exp, x0)


    try
        [t_sim, y_raw] = ode45(@(t,y) trueODEfunc_multiguild(t,y,p), t_exp, x0);
        %{
          t değişkeni :
            ode45 simülasyonu yaparken zamanı milim milim ilerletir.
            t, o anki hesaplama adımındaki saattir. (Örn: 0.0001. saniye).
            for(int i=0; ...) döngüsündeki i gibidir. Sadece döngü içinde yaşar.
        %}
         %{
          y değişkeni :
          (Current State - Anlık Durum):
          O anki t anında sistemin durumudur.
          10 elemanlı bir vektördür: [H2, CO2, ..., Bakteri, Asetat].
          Motor, trueODEfunc fonksiyonuna şunu sorar: "Zaman t iken, elimdeki kimyasallar y ise, bir sonraki saniyede değişim ne olur?"
        %}
          %{
          p değişkeni :
            p (Parametreler / Kurallar)
            Nedir?: lsqnonlin tarafından gönderilen deneme tahtasıdır.
            İçeriği: [2.5, 0.1, 0.05 ...] gibi 13 adet sayıdır.
            Görevi: trueODEfunc fonksiyonuna kuralları söyler. "Bakteri yeme hızını şimdilik 2.5 kabul et" der.
          %}     


          %{
          t_sim (Simülasyon Zaman Çizelgesi)
            Nedir?: Motorun işi bittiğinde bize verdiği raporun Zaman Sütunu.
            Yapısı: Nx1 boyutunda bir vektör (Örn: 500 satır).
            Önemli Detay (Variable Step): ode45 zekidir. Reaksiyonlar hızlıysa zamanı 0.001 saniye ilerletir, yavaşsa 0.5 saniye atlar.
            Bu yüzden t_sim düzgün artmaz: [0, 0.001, 0.005, 0.1, 0.15 ...] gibi gider. Laboratuvar günleriyle (1.1, 5.0) uyuşmaz!
          %}
            
         %{
          y_raw (Ham Simülasyon Sonucu)
          Nedir?: t_sim zamanlarındaki kimyasal değerler.
          Yapısı: Nx10 boyutunda bir matris.
          Anlamı: "Simülasyonun 5. satırındaki t_sim anında, H2 miktarı y_raw(5,1) kadardı."
          Neden "Raw"?: Çünkü zaman adımları düzensizdir (adaptive step). Bu veriyi doğrudan laboratuvar verisiyle kıyaslayamayız, çünkü zamanlar denk gelmiyor.
          %}
        


        y_sim = interp1(t_sim, y_raw, t_exp, 'linear'); %Interpolasyon 


        %{        
        1. interp1 (Zaman Senkronizasyonu)
    Sorun: Simülasyonun saati ile laboratuvarın saati tutmuyor. Simülasyon motoru (ode45), olayların hızlı olduğu yerlerde milisaniyelik adımlar atarken, yavaş yerlerde saatlik adımlar atabilir.
    Simülasyonun Adımları (t_sim): 1.09. gün, 1.12. gün, 1.15. gün...
    Gerçek Veri (t_exp): Tam 1.10. gün.
    Bilgisayar çıkarma işlemi yapamaz: Simülasyon(1.12) - Gerçek(1.10) = Hata! Zamanlar aynı değil.
    Çözüm (interp1): Bu fonksiyon, simülasyon noktalarının arasına hayali bir çizgi çeker ve tam istediğimiz saniyedeki değeri hesaplar.        
       
        
        Mantık: "Eğer 1.09. günde H2 seviyesi 100 ise, 1.12. günde 90 ise; tam 1.10. günde (arada bir yerde) yaklaşık 97 olmalıdır."
        Kod: y_sim = interp1(t_sim, y_raw, t_exp, 'linear'); 
        Simülasyon verisini (y_raw), gerçek deney zamanlarına (t_exp) hizalar. Artık elimizde tam 1.1. güne ait yapay bir veri var.
        
        %}

        log_sim = log1p(y_sim(:,1:5)); % H2, CO2, CH4, H2S, SO4

          %{        
         log1p (Teraziyi Eşitleme / Logaritma)Sorun: Adalet Terazisi Bozuk.Verimizde çok büyük ve çok küçük sayılar yan yana duruyor.
H2 (Hidrojen): 9000 birim.H2S (Sülfür): 0.001 birim.Dedektif hatayı hesaplarken farka bakar (Simülasyon - Gerçek):H2'de %10 hata yaparsa: $9000 - 8100 = \mathbf{900}$ puan ceza.H2S'de %10 hata yaparsa: $0.001 - 0.0009 = \mathbf{0.0001}$ puan ceza.Sonuç: Dedektif der ki: "H2S'yi boşver, o mikrop kadar küçük. Bütün gücümü H2'yi düzeltmeye harcayayım." Ama biyolojik olarak H2S çok önemlidir (zehirlidir)!Çözüm (log1p):Sayıları Logaritmik Ölçeğe çekerek hepsini aynı ligde oynatırız.
 Logaritma, büyük sayıları ezer, küçük sayıları parlatır.$\log(9000) \approx 9.1$$\log(0.001) \approx -6.9$Bakın, artık ikisi de "Tek haneli" sayılar oldu! Dedektif artık H2S'deki hatayı da ciddiye alacaktır.
Neden 1p (Plus 1)?Matematikte $\log(0)$ tanımsızdır ($-\infty$).Eğer bir kimyasal biterse (0 olursa), kod çöker.log1p(x) aslında $\ln(1+x)$ demektir. $x=0$ olsa bile $\ln(1)=0$ olur, kod güvenle çalışır2.
          %}
        log_exp = log1p(data_exp);
        weights = [1, 1, 0.5, 0.5, 1];
        res = (log_sim - log_exp) .* weights;

        if any(y_sim(:) < -1e-6)
            res = res + 1e3 * abs(min(y_sim(:)));
        end
        res = res(:);
    catch
        res = 1e6 * ones(numel(data_exp), 1);
    end
end
%{
Büyük Resim: Analiz Nasıl Yapılıyor?
Kodun 82-84. satırlarında analiz şöyle akar:
Hizala (interp1): Önce simülasyon saatlerini gerçek saatlere uydur.
Ölçekle (log1p): Sayıların logaritmasını al ki, küçük kimyasallar ezilmesin.
Farkı Al (Subtract): Log(Simülasyon) - Log(Gerçek).
Ağırlıklandır (weights): Bazı kimyasalları (H2, CO2) daha önemli say (weights vektörü ile çarp).
Bu işlemler sonucunda tek bir "Hata Puanı" (res) çıkar. lsqnonlin motoru da bu puanı düşürmeye çalışır.
%}


%% Mechanistic ODE function for multi-guild system
function dydt = trueODEfunc_multiguild(~, y, p)
% State variables
    H2 = y(1); CO2 = y(2); CH4 = y(3); H2S = y(4);
    SO4 = y(5); FeS = y(6); X_meth = y(7); X_sulf = y(8); X_aceto = y(9); Acetate = y(10);
 % Parameters
 %  Maximum rates
    k_meth = p(1);  % methanogenesis reaction
    k_sulf = p(2);  % sulfate reduction reaction
    k_aceto = p(3);  % acetogenesis reaction
    % Biomass yields
    Y_m = p(4);   % biomas yields from methanogens
    Y_s = p(5);   % from sulfate reduction
    Y_a = p(6);   % from acetogens
    % Inhibition constants
    KI_meth = p(7);  
    KI_sulf = p(8);
    KI_aceto = p(9);
    % Precipitation rate and saturation
    k_precip = p(10);
    H2S_sat = p(11);
    H2_thresh = p(12);
    % ΔG threshold 
    DG_thresh = p(13);
 % Constants -- Dynamic ΔG via Nernst Equation
    R = 8.314e-3; T = 298.15; RT = R*T;
    DG0_meth = -130; DG0_sulf = -152; DG0_aceto = -95;
  % Inhibition
    f_inh_meth  = KI_meth  / (KI_meth  + H2S); % inhibition by H₂S
    f_inh_sulf  = KI_sulf  / (KI_sulf  + H2S);
    f_inh_aceto = KI_aceto / (KI_aceto + H2S);
     % Activation threshold
    f_activation = H2 / (H2 + H2_thresh); % low-H₂ suppression
% Avoid log of zero or negative   % Thermodynamic feasibility
    H2 = max(H2,1e-6); CO2 = max(CO2,1e-6); CH4 = max(CH4,1e-6);
    SO4 = max(SO4,1e-6); H2S = max(H2S,1e-6); Acetate = max(Acetate,1e-6);
% Reaction quotients
    Q_meth  = CH4     / (H2^4 * CO2);
    Q_sulf  = H2S     / (H2^4 * SO4);
    Q_aceto = Acetate / (H2^4 * CO2^2);
% Dynamic Gibbs energies
    DG_meth  = DG0_meth  + RT*log(Q_meth);
    DG_sulf  = DG0_sulf  + RT*log(Q_sulf);
    DG_aceto = DG0_aceto + RT*log(Q_aceto);
% Thermodynamic feasibility
    f_thermo_meth  = 1 / (1 + exp((DG_meth  - DG_thresh)/RT));
    f_thermo_sulf  = 1 / (1 + exp((DG_sulf  - DG_thresh)/RT));
    f_thermo_aceto = 1 / (1 + exp((DG_aceto - DG_thresh)/RT));
   % Reaction rates with thermodynamic scaling
    r_meth  = k_meth  * H2 * CO2^(-2) * f_inh_meth  * f_activation * f_thermo_meth;
    r_sulf  = k_sulf  * H2 * SO4      * f_inh_sulf  * f_activation * f_thermo_sulf;
    r_aceto = k_aceto * H2 * CO2^2    * f_inh_aceto * f_activation * f_thermo_aceto;
    r_precip = k_precip * max(0, H2S - H2S_sat);

    % Differential equations
    dH2      = -4*r_meth - 4*r_sulf - 4*r_aceto;
    dCO2     = -1*r_meth - 2*r_aceto;
    dCH4     = +1*r_meth;
    dH2S     = +1*r_sulf - r_precip;
    dSO4     = -1*r_sulf;
    dFeS     = +1*r_precip;
    dX_meth  = Y_m * r_meth;
    dX_sulf  = Y_s * r_sulf;
    dX_aceto = Y_a * r_aceto;
    dAcetate = +1*r_aceto;

    dydt = [dH2; dCO2; dCH4; dH2S; dSO4; dFeS; dX_meth; dX_sulf; dX_aceto; dAcetate];
end