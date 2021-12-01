function plot_S3yrprarbrr_s1yrpraobro(species, ys, ts)

s0 = find(contains(species,'S3YrPrArBrr'));
s1 = find(contains(species,'S1YrPrAoBro'));
s2 = find(strcmp(species,'S3YrPrAr'));





S0 = [];
S1 =[];
S2 =[];
t = [];  


for i  = 2:length(ys)
    S0 = [S0, ys{i}(s0,:)];
    S1 = [S1, ys{i}(s1,:)];
    S2 = [S2, ys{i}(s2,:)];
    t = [t,ts{i}];    
end
figure;
hold on
plot(t, S0, 'k', 'linewidth',1.2)
plot(t, S1, 'r', 'linewidth',1.2)
plot(t, S2, 'b', 'linewidth',1.2)

ylabel('Conc ')
xlabel('time, s')
legend({'S3YrPrArBrr','S1YrPrAoBro', 'S3YrPrAr'})