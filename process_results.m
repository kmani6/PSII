function process_results(analysis_name)

results_dir = ['results/', analysis_name];
results = ls(results_dir);
nresults = size(results);
nresults = nresults(1);

fval_opt = inf;
for i = 3:nresults
    fprintf(results(i,:))
    fprintf('\n')
    %result_name = [analysis_name, ]
    
    
    
end

foo = 1;


end
