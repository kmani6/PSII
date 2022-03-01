function plot_S1_states(species, ys, ts)

s4_all = find(contains(species,'S1'));
figure;
counter = 0;
for i = 1:4
    s4 = s4_all((i-1)*6+1:(i-1)*6+6);
    counter = counter+6;
    subplot(2,2,i)
    S4 =[];
    t = [];  


    for i  = 2:length(ys)
        S4 = [S4, (ys{i}(s4,:))];
        t = [t,ts{i}];    
    end
    plot(t, S4,'linewidth',1.2)
    ylabel('S-state fraction')
    xlabel('time, s')
    legend(species(s4))

end
end