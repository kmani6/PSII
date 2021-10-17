function f = plot_pq_redox_state(species, ys, ts) % model specific variables)

pq = strcmp(species,'PQ');
pqh2 = strcmp(species,'PQH2');
boo = find(contains(species,'Boo'));
bro = find(contains(species,'Bro'));
brr = find(contains(species,'Brr'));

f = figure;
t = [];
PQ = [];
PQH2 = [];
BOO = [];
BRO = [];
BRR = [];

for itime = 2:length(ys)
    plot(ts{itime},sum(ys{itime}(boo,:)))
    PQ = [PQ, ys{itime}(pq,:)];
    PQH2 = [PQH2, ys{itime}(pqh2,:)];
    BOO = [BOO, sum(ys{itime}(boo,:))];
    BRO = [BRO, sum(ys{itime}(bro,:))];
    BRR = [BRR, sum(ys{itime}(brr,:))];
    t = [t, ts{itime}];
end
plot(t, PQH2./(PQ+PQH2))

title('PQ pool redox state')
ylabel('reduced PQ fraction')
xlabel('time')


figure
hold on
plot(t, PQ);
plot(t, PQH2);
plot(t, BOO);
plot(t, BRO);
plot(t, BRR);
legend({'PQ','PQH2','Boo','Bro','Brr'})
title('PQ species')
ylabel('concentration')
xlabel('time')

    
    

