function plot_reaction_rates(indices, rs, ts)

rates = [];
t = [];
counter_time = 1;
for i = 2:length(rs)
    counter_index = 1;
    t = [t, ts{i}];
    rates = [ rates, rs{i}(indices, :)];
        
        
 
    


end

figure;
xlabel = 'time';
ylabel = 'reaction rate';
legends = {};
hold on
for i = 1:length(indices)
    plot(t, rates(i,:), 'LineWidth',1)

    legends{end+1} = num2str(indices(i));
    
end
legend(legends)


end
