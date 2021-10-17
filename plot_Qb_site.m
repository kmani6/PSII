function plot_Qb_site(species, ys, ts)

boo = find(contains(species,'Boo'));
bro = find(contains(species,'Bro'));
brr = find(contains(species,'Brr'));
empty = find(~contains(species,'B') & contains(species,'S'));




YOPOAX = [];
BOO = [];
BRO =[];
BRR = [];
EMPTY = [];
t = [];  


for i  = 2:length(ys)
    BOO = [BOO, sum(ys{i}(boo,:))];
    BRO = [BRO, sum(ys{i}(bro,:))];
    BRR = [BRR, sum(ys{i}(brr,:))];
    EMPTY = [EMPTY, sum(ys{i}(empty,:))];
    t = [t,ts{i}];    
end
figure;
hold on
plot(t, EMPTY)
plot(t, BOO)
plot(t, BRO)
plot(t, BRR)
legend({'Empty','Boo','Bro','Brr'})