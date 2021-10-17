function plot_QaQbBxx_site_NEW(species, ys, ts)

S2YoPoArBrr = find(contains(species,'S2YoPoArBrr'));
S3YrPrAoBrr = find(contains(species,'S3YrPrAoBrr'));
S0YrPrArBrr = find(contains(species,'S0YrPrArBrr'));
S3YoPrArBrr = find(contains(species,'S3YoPrArBrr'));
S2YrPrAoBro = find(contains(species,'S2YrPrAoBro'));
S2YoPrArBro = find(contains(species,'S2YoPrArBro'));
S3YoPrAr = find(~contains(species,'B') & contains(species,'S3YoPrAr'));

S1YoPoArBrr = find(contains(species,'S1YoPoArBrr'));
S2YrPrAoBrr = find(contains(species,'S2YrPrAoBrr'));
S3YrPrArBrr = find(contains(species,'S3YrPrArBrr'));
S2YoPrAr = find(~contains(species,'B') & contains(species,'S2YoPrAr'));
S1YoPrArBro = find(contains(species,'S1YoPrArBro'));
S1YrPrAoBro = find(contains(species,'S1YrPrAoBro'));

S2YOPOARBRR = [];
S3YRPRAOBRR = [];
S0YRPRARBRR = [];
S3YOPRARBRR = [];
S2YRPRAOBRO = [];
S2YOPRARBRO = [];
S3YOPRAR = [];

S1YOPOARBRR = [];
S2YRPRAOBRR = [];
S3YRPRARBRR = [];
S2YOPRAR = [];
S1YOPRARBRO = [];
S1YRPRAOBRO = [];

t = [];  


for i  = 2:length(ys)
    
    S2YOPOARBRR = [S2YOPOARBRR, (ys{i}(S2YoPoArBrr,:))];
    S3YRPRAOBRR = [S3YRPRAOBRR, (ys{i}(S3YrPrAoBrr,:))];
    S0YRPRARBRR = [S0YRPRARBRR, (ys{i}(S0YrPrArBrr,:))];
    S3YOPRARBRR = [S3YOPRARBRR, (ys{i}(S3YoPrArBrr,:))];
    S2YRPRAOBRO = [S2YRPRAOBRO, (ys{i}(S2YrPrAoBro,:))];
    S2YOPRARBRO = [S2YOPRARBRO, (ys{i}(S2YoPrArBro,:))];
    S3YOPRAR = [S3YOPRAR, (ys{i}(S3YoPrAr,:))];
    
    S1YOPOARBRR = [S1YOPOARBRR, (ys{i}(S1YoPoArBrr,:))];
    S2YRPRAOBRR = [S2YRPRAOBRR, (ys{i}(S2YrPrAoBrr,:))];
    S3YRPRARBRR = [S3YRPRARBRR, (ys{i}(S3YrPrArBrr,:))];
    S2YOPRAR = [S2YOPRAR, (ys{i}(S2YoPrAr,:))];
    S1YOPRARBRO = [S1YOPRARBRO, (ys{i}(S1YoPrArBro,:))];
    S1YRPRAOBRO = [S1YRPRAOBRO, (ys{i}(S1YrPrAoBro,:))];
    
    t = [t,ts{i}]; 
    
end

figure;
hold on
plot(t, S2YOPOARBRR,'LineWidth',2.0)
plot(t, S3YRPRAOBRR,'LineWidth',2.0)
plot(t, S0YRPRARBRR,'LineWidth',2.0)
plot(t, S3YOPRARBRR,'LineWidth',2.0)
plot(t, S2YRPRAOBRO,'LineWidth',2.0)
plot(t, S2YOPRARBRO,'LineWidth',2.0)
plot(t, S3YOPRAR,'LineWidth',2.0)

legend({'S2YOPOARBRR','S3YRPRAOBRR','S0YRPRARBRR','S3YOPRARBRR','S2YRPRAOBRO','S2YOPRARBRO','S3YOPRAR'})

figure;
hold on
plot(t, S1YOPOARBRR,'LineWidth',2.0)
plot(t, S2YRPRAOBRR,'LineWidth',2.0)
plot(t, S3YRPRARBRR,'LineWidth',2.0)
plot(t, S2YOPRAR,'LineWidth',2.0)
plot(t, S1YOPRARBRO,'LineWidth',2.0)
plot(t, S1YRPRAOBRO,'LineWidth',2.0)

legend({'S1YOPOARBRR','S2YRPRAOBRR','S3YRPRARBRR','S2YOPRAR','S1YOPRARBRO','S1YRPRAOBRO'})