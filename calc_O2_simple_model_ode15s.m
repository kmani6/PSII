function [O2] = calc_O2_simple_model_ode15s(x0, n_trains, n_flashes, ...
                    flash_duration, flash_interval, train_interval, ... 
                    Fluorescence_k_idcs, Fluorescence_y_inds, kidcs, ... 
                    tablek, tabley, kconst, rate_inds, ... 
                    S, species, species_idcs, Rknames)
                    
                
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

mult1 = kidcs.mult1;
mult2 = kidcs.mult2;
n1idx = kidcs.n1idx;
ts = {};
rs = {};
O2 =  zeros(n_trains*n_flashes,1);
O2_light =  zeros(n_trains*n_flashes,1);
O2_dark =  zeros(n_trains*n_flashes,1);
O2ind = find(strcmp(species, 'O2'));
% dark adapt the system
k(mult1) = 0;
k(mult2) = 0;    
k(n1idx) = 0;
dark_adaptation_time = 1; 
t_lims = [0,dark_adaptation_time];
Sol =  ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t_lims,yinitial);
ts{end+1} = -dark_adaptation_time+Sol.x;
yinitial = Sol.y(:,end); 
counter = 1;
for train = 1:n_trains
    if train == 9
        foo = 1;
    end
    for flash = 1:n_flashes
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
        ts{end+1} = ts{end}(end) + Sol.x;

        flash_O2 = Sol.y(O2ind,:) -Sol.y(O2ind,1);
        t_O2 = Sol.x;
%         O2_light(counter) = trapz(Sol.x, flash_O2);

        yinitial = Sol.y(:,end); 
        if any(isnan(yinitial))
            foo = 1;
        end

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
            ts{end+1} = ts{end}(end) + Sol.x;
            yinitial = Sol.y(:,end);
        end
        O2_dark_prod = Sol.y(O2ind,:) - Sol.y(O2ind,1);
        O2_dark_prod = flash_O2(end) + O2_dark_prod;
        O2_total_prod = [flash_O2, O2_dark_prod];
        t_O2 = [t_O2, Sol.x];
%         O2_dark(counter) = trapz(Sol.x, O2_dark_prod);
%         O2(counter) = O2(counter)+O2_dark(counter);
        O2(counter) = O2_total_prod(end);
        counter = counter+1;

    end
    k(mult1) = 0;
    k(mult2) = 0;       
    k(n1idx) = 0;
    t = linspace(0, train_interval, train_interval*1e5);
    Sol = ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);

        if any(any(Sol.y<-1e-5)) || any(any(isnan(Sol.y)))
                nTimepoints = train_interval*1e3;
                t = linspace(0, train_interval, nTimepoints);
                Sol = ode2(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);    
                while any(any(Sol<-1e-5)) || any(any(isnan(Sol)))
                    nTimepoints = nTimepoints*5;
                    t = linspace(0, train_interval, nTimepoints);
                    Sol = ode2(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t,yinitial);
                end
            Sol = Sol';
            ts{end+1} = ts{end}(end) + t;
            yinitial = Sol(:,end);
        else
            ts{end+1} = ts{end}(end) + Sol.x;
            yinitial = Sol.y(:,end);
        end

end
end
