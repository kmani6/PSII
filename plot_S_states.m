function plot_S_states(species, ys, ts)

s0 = find(contains(species,'S0'));
s1 = find(contains(species,'S1'));
s2 = find(contains(species,'S2'));
s3 = find(contains(species,'S3'));




YOPOAX = [];
S0 = [];
S1 =[];
S2 = [];
S3 = [];
t = [];  


for i  = 2:length(ys)
    S0 = [S0, sum(ys{i}(s0,:))];
    S1 = [S1, sum(ys{i}(s1,:))];
    S2 = [S2, sum(ys{i}(s2,:))];
    S3 = [S3, sum(ys{i}(s3,:))];
    t = [t,ts{i}];    
end
figure;
PSII_tot = S0(1)+S1(1)+S2(1)+S3(1);
hold on
plot(t, S0./PSII_tot, 'k', 'linewidth',1.2)
plot(t, S1./PSII_tot, 'r', 'linewidth',1.2)
plot(t, S2./PSII_tot, 'b', 'linewidth',1.2)
plot(t, S3./PSII_tot, 'linewidth',1.2, 'color', [0,0.5,0])
ylabel('S-state fraction')
xlabel('time, s')
legend({'S0','S1','S2','S3'})