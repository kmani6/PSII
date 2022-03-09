function [sqerror, dsqerror] = calc_sqerror_simple_model_O2(x,... Set of parameters. This only includes the independent variables as described by the third column in Y and Constants files
                    n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... Experimental parameters
                    Fluorescence_k_idcs, Fluorescence_y_inds,... Indeces to calculate Fluorescence
                    kidcs, PSIidcs, ... all indices needed in to calculate FvFm and prepare the variables
                    tablek, tabley,... information on the k and y variables
                    kconst, rate_inds, S, species, knames, Rknames, species_idcs,... model specific variables
                    FvFm_exp,O2_exp)
                
try
                
[O2_sim] = calc_O2_simple_model_ode15s( x, ...
                n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... 
                Fluorescence_k_idcs, Fluorescence_y_inds,...
                kidcs, tablek, tabley, kconst, rate_inds, ... 
                S, species, species_idcs, Rknames); % model specific variables
                    
                
             


% x,... Set of parameters. This only includes the independent variables as described by the third column in Y and Constants files
%                     n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... Experimental parameters
%                     Fluorescence_k_idcs, Fluorescence_y_inds,... Indeces to calculate Fluorescence
%                     kidcs, PSIidcs, ... all indices needed in to calculate FvFm and prepare the variables
%                     tablek, tabley,... information on the k and y variables
%                     kconst, rate_inds, S, species, knames, species_idcs); % model specific variables

O2_length = length(O2_sim);
O2_avg = sum(O2_sim)/(O2_length);  
O2_sim = (O2_sim/O2_avg);
sqerror = norm(O2_exp-O2_sim)^2/numel(O2_exp); %+ 1e-10* norm(grads)/numel(grads);

catch err
  disp(getReport(err));
    
    sqerror = nan;

    
end


if nargout ==2
    mu = 1e-12;
    U = normrnd(0,1,length(x),1);
    x_tmp = x+mu*U;
    try
    [O2_tmp] = calc_O2_simple_model_ode15s(x_tmp,... Set of parameters. This only includes the independent variables as described by the third column in Y and Constants files
                    n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... Experimental parameters
                    Fluorescence_k_idcs, Fluorescence_y_inds,... indeces used to calculate fluorescence
                    kidcs, PSIidcs, ... all indices needed in to calculate FvFm and prepare the variables
                    tablek, tabley,... information on the k and y variables
                    kconst, rate_inds, S, species, knames, species_idcs, Rknames); % model specific variables
                
    sqerror_tmp = norm(O2_exp-O2_tmp)^2/numel(O2_exp);
    dsqerror = -(sqerror_tmp-sqerror)/mu*U/10;
    catch err
    disp(getReport(err));
    
    dsqerror = nan;
    end
                
end
