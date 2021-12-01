function plot_QaQbBxx_site(species, ys, ts)

S3YoPrArBrr = find(contains(species,'S3YoPrArBrr'));
S3YoPrArBoo = find(contains(species,'S3YoPrArBoo'));
S3YoPrAoBro = find(contains(species,'S3YoPrAoBro'));
S0YrPrArBrr = find(contains(species,'S0YrPrArBrr'));
S1YrPrAoBro = find(contains(species,'S1YrPrAoBro'));
S2YrPrAoBro = find(contains(species,'S2YrPrAoBro'));
emptyS3 = find(~contains(species,'B') & contains(species,'S3YoPrAr'));

S2YoPrArBrr = find(contains(species,'S2YoPrArBrr'));
S2YoPrArBoo = find(contains(species,'S2YoPrArBoo'));
S2YoPrAoBro = find(contains(species,'S2YoPrAoBro'));
emptyS2 = find(~contains(species,'B') & contains(species,'S2YoPrAr'));
S3YrPrArBrr = find(contains(species,'S3YrPrArBrr'));
S1YrPrAoBro = find(contains(species,'S2YrPrAoBro'));%


S3YOPRARBRR = [];
S3YOPRARBOO = [];
S3YOPRAOBRO = [];
S0YRPRARBRR = [];
S1YRPRAOBRO = [];
S2YRPRAOBRO = [];
EMPTYS3 = [];

S2YOPRARBRR = [];
S2YOPRARBOO = [];
S2YOPRAOBRO = [];
EMPTYS2 = [];
S3YRPRARBRR = [];
S1YRPRAOBRO = [];

t = [];  


for i  = 2:length(ys)
    
    S3YOPRARBRR = [S3YOPRARBRR, (ys{i}(S3YoPrArBrr,:))];
    S3YOPRARBOO = [S3YOPRARBOO, (ys{i}(S3YoPrArBoo,:))];
    S3YOPRAOBRO = [S3YOPRAOBRO, (ys{i}(S3YoPrAoBro,:))];
    S0YRPRARBRR = [S0YRPRARBRR, (ys{i}(S0YrPrArBrr,:))];
    S1YRPRAOBRO = [S1YRPRAOBRO, (ys{i}(S1YrPrAoBro,:))];
    EMPTYS3 = [EMPTYS3, (ys{i}(emptyS3,:))];
    
    S2YOPRARBRR = [S2YOPRARBRR, (ys{i}(S2YoPrArBrr,:))];
    S2YOPRARBOO = [S2YOPRARBOO, (ys{i}(S2YoPrArBoo,:))];
    S2YOPRAOBRO = [S2YOPRAOBRO, (ys{i}(S2YoPrAoBro,:))];
    EMPTYS2 = [EMPTYS2, (ys{i}(emptyS2,:))];
    S3YRPRARBRR = [S3YRPRARBRR, (ys{i}(S3YrPrArBrr,:))];
    %S1YRPRAOBRO = [S1YRPRAOBRO, (ys{i}(S2YrPrAoBro,:))];
    
    t = [t,ts{i}];    
end
figure;
hold on
plot(t, S3YOPRARBRR,'LineWidth',2.0)
plot(t, S3YOPRARBOO,'LineWidth',2.0)
plot(t, S3YOPRAOBRO,'LineWidth',2.0)
plot(t, S0YRPRARBRR,'LineWidth',2.0)
plot(t, S1YRPRAOBRO,'LineWidth',2.0)
plot(t, EMPTYS3,'LineWidth',2.0)

legend({'S3YOPRARBRR','S3YOPRARBOO','S3YOPRAOBRO','S0YRPRARBRR','S1YRPRAOBRO','EMPTYS3'})

figure;
hold on
plot(t, S2YOPRARBRR,'LineWidth',2.0)
plot(t, S2YOPRARBOO,'LineWidth',2.0)
plot(t, S2YOPRAOBRO,'LineWidth',2.0)
plot(t, EMPTYS2,'LineWidth',2.0)
plot(t, S3YRPRARBRR,'LineWidth',2.0)
plot(t, S1YRPRAOBRO,'LineWidth',2.0)

legend({'S2YOPRARBRR','S2YOPRARBOO','S2YOPRAOBRO','EMPTYS2','S3YRPRARBRR','S1YRPRAOBRO'})