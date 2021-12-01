function plot_S3_S1_states(species, ys, ts)

s1 = find(contains(species,'S1'));
s3 = find(contains(species,'S3'));




S1 =[];
S3 = [];
t = [];  


for i  = 2:length(ys)
    S1 = [S1, sum(ys{i}(s1,:))];
    S3 = [S3, sum(ys{i}(s3,:))];
    t = [t,ts{i}];    
end
figure;
hold on
plot(t, S1, 'k', 'linewidth',1.2)
plot(t, S3, 'linewidth',1.2, 'color', [0,0.5,0])
ylabel('S-state fraction')
xlabel('time, s')
legend({'S0','S3'})