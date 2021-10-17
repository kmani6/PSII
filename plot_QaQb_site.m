function plot_QaQb_site(species, ys, ts)

arboo = find(contains(species,'ArBoo'));
arbro = find(contains(species,'ArBro'));
arbrr = find(contains(species,'ArBrr'));
aoboo = find(contains(species,'AoBoo'));
aobro = find(contains(species,'AoBro'));
aobrr = find(contains(species,'AoBrr'));
empty = find(~contains(species,'B') & contains(species,'S'));




YOPOAX = [];
ARBOO = [];
ARBRO =[];
ARBRR = [];
AOBOO = [];
AOBRO =[];
AOBRR = [];
EMPTY = [];
t = [];  


for i  = 2:length(ys)
    ARBOO = [ARBOO, sum(ys{i}(arboo,:))];
    ARBRO = [ARBRO, sum(ys{i}(arbro,:))];
    ARBRR = [ARBRR, sum(ys{i}(arbrr,:))];
    AOBOO = [AOBOO, sum(ys{i}(aoboo,:))];
    AOBRO = [AOBRO, sum(ys{i}(aobro,:))];
    AOBRR = [AOBRR, sum(ys{i}(aobrr,:))];
    EMPTY = [EMPTY, sum(ys{i}(empty,:))];
    t = [t,ts{i}];    
end
figure;
hold on
plot(t, EMPTY,'LineWidth',2.0)
plot(t, ARBOO,'LineWidth',2.0)
plot(t, ARBRO,'LineWidth',2.0)
plot(t, ARBRR,'LineWidth',2.0)
plot(t, AOBOO,'LineWidth',2.0)
plot(t, AOBRO,'LineWidth',2.0)
plot(t, AOBRR,'LineWidth',2.0)
legend({'Empty','ARBOO','ARBRO','ARBRR','AOBOO','AOBRO','AOBRR'})