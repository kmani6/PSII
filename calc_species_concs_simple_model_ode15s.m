function [ts,ys,FvFm] = calc_species_concs_simple_model_ode15s(x0,... Set of parameters. This only includes the independent variables as described by the third column in Y and Constants files
                    n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... Experimental parameters
                    Fluorescence_k_idcs, Fluorescence_y_inds,... indeces used to calculate fluorescence
                    kidcs, PSIidcs, ... all indices needed in to calculate FvFm and prepare the variables
                    tablek, tabley,... information on the k and y variables
                    kconst, rate_inds, S, species, knames, species_idcs, Rknames) % model specific variables
                    

                
indepy = find(tabley.independent);
yinitial = tabley.base_val;
yr = x0(1:length(indepy));
yinitial(indepy) = yr;
yinitial = yinitial(species_idcs);

indepk = find(tablek.independent);
kr = x0(length(indepy)+1:end);
k = tablek.base_val;
k(indepk) = kr;



k(kidcs.kf) = 1;
PS1T = (1-k(kidcs.a2))*k(kidcs.Chl)/k(kidcs.PSU1); 
PS2T = k(kidcs.a2)*(k(kidcs.Chl)/k(kidcs.PSU2));
n1 = k(kidcs.PFD)*k(kidcs.Labs)*(1-k(kidcs.a2))/PS1T;
n2 = k(kidcs.PFD)*k(kidcs.Labs)*k(kidcs.a2)/PS2T;                 
% yinitial(PSIidcs) = PS1T/PS2T;    %- Commented out for the simpler model
% up to PQ
     
mult1Val = n2*k(kidcs.kp)/(1+k(kidcs.kp)+k(kidcs.kn)+k(kidcs.kr));
mult2Val = n2*k(kidcs.kp)/(1+k(kidcs.kp)+k(kidcs.kn));
Div1Val = k(kidcs.kpc)/k(kidcs.kEpc);
Div2Val = k(kidcs.kcytf)/k(kidcs.kEcytf);
Div3Val = k(kidcs.kfx)/k(kidcs.kEfx);
Div4Val = k(kidcs.kb6f)/k(kidcs.kEb6f);

k(kidcs.mult1) = mult1Val;
k(kidcs.mult2) = mult2Val;
k(kidcs.Div1) = Div1Val;
k(kidcs.Div2) = Div2Val;
k(kidcs.Div3) = Div3Val; 
k(kidcs.Div4) = Div4Val;
k(kidcs.n1idx) = n1;
FvFm = zeros(n_trains*n_flashes,1);

mult1 = kidcs.mult1;
mult2 = kidcs.mult2;
n1idx = kidcs.n1idx;
% dark adapt the system
k(mult1) = 0;
k(mult2) = 0;    
k(n1idx) = 0;
dark_adaptation_time = 1e-07; %3-5 minutes typically
t_lims = [0,dark_adaptation_time];
Sol =  ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t_lims,yinitial);
% ts{end+1} = -dark_adaptation_time+Sol.x;
% ys{end+1} = Sol.y;
yinitial = Sol.y(:,end); %initialize the y vector for the next iteration 
counter = 1;

ts = {};
ys = {};
Fs = {};

ts{end+1} = 0;%-dark_adaptation_time+Sol.x;
ys{end+1} = 0;%Sol.y;
Fs{end+1} = [];
for train = 1:n_trains
%     fprintf('train %i \n', train)
    if train == 9
        foo = 1;
    end
    for flash = 1:n_flashes
%         fprintf('flash %i \n', flash)
        k(mult1) = mult1Val;
        k(mult2) = mult2Val;    
        k(n1idx) = n1;
        nTimepoints = flash_duration*1e8;
        t = [0, flash_duration];
        Sol = ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);
        while any(any(Sol.y<-1e-5)) || any(any(isnan(Sol.y)))
            nTimepoints = nTimepoints*5;
            t = linspace(0, flash_duration, nTimepoints);
            Sol = ode2(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);    
            
        end
        %F = LaiskFluorescence(Fluorescence_y_inds, Fluorescence_k_idcs, k, Sol.y) ;
        %F1 = F;
        t1 = Sol.x;
        %FvFm(counter) = (max(F) - F(1))/max(F);
        ts{end+1} = ts{end}(end) + Sol.x;
        ys{end+1} = Sol.y;
        yinitial = Sol.y(:,end); %initialize the y vector for the next iteration 
        if any(isnan(yinitial))
            foo = 1;
        end

        %Shift to dark time between flashes
        k(mult1) = 0;
        k(mult2) = 0;      
        k(n1idx) = 0;
        
        t_lims = [0,flash_interval];
        Sol = ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t_lims,yinitial);
        if any(any(Sol.y<-1e-5)) || any(any(isnan(Sol.y)))
            nTimepoints = flash_interval*1e3;
            t = linspace(0, flash_interval, nTimepoints); 
            Sol = ode2(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);    
            while any(any(Sol<-1e-5)) || any(any(isnan(Sol)))
                nTimepoints = nTimepoints*5;
                t = linspace(0, flash_interval, nTimepoints);
                Sol = ode2(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);    
            end
            Sol = Sol';
            ts{end+1} = ts{end}(end) + t;
            yinitial = Sol(:,end);
        else
            yinitial = Sol.y(:,end);
            ts{end+1} = ts{end}(end) + Sol.x;
            ys{end+1} = Sol.y;
        end
        counter = counter+1;
    end
    k(mult1) = 0;
    k(mult2) = 0;       
    k(n1idx) = 0;
    t = linspace(0, train_interval, train_interval*1e5);
    %t = linspace(0, train_interval, train_interval*1e12);
    Sol = ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);
        if any(any(Sol.y<-1e-5)) || any(any(isnan(Sol.y)))
                nTimepoints = train_interval*1e3;
                %nTimepoints = train_interval*1e10;
                t = linspace(0, train_interval, nTimepoints);
                Sol = ode2(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);    
                while any(any(Sol<-1e-5)) || any(any(isnan(Sol)))
                    nTimepoints = nTimepoints*5;
                    t = linspace(0, train_interval, nTimepoints);
                    Sol = ode2(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);
                end
            Sol = Sol';
            yinitial = Sol(:,end);
        else

            yinitial = Sol.y(:,end);
            ts{end+1} = ts{end}(end) + Sol.x;
            ys{end+1} = Sol.y;
        end
end
end
