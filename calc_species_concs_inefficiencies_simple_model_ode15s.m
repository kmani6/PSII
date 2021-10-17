function [ts,ys,FvFm,y_totals,P_totals] = calc_species_concs_inefficiencies_simple_model_ode15s(x0,... Set of parameters. This only includes the independent variables as described by the third column in Y and Constants files
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
% yinitial(PSIidcs) = PS1T/PS2T;    %- Commented out for the simpler model up to PQ
     
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
dark_adaptation_time = 1; %3-5 minutes typically
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
figure;
pat = "S";
PS2 = startsWith(species,pat);
species2 = species(PS2);



B = 262;
X = zeros(1,B);
Y = zeros(1,B);


S2 = S(PS2,:);
for i = 1:B
    
    v = S2(:,i);
    
    ninds = find(v == -1);
    pinds = find(v == 1);
    
    nnames = species2(ninds);
    pnames = species2(pinds);
    
    pat = "S";
    nTF = startsWith(nnames,pat);
    pTF = startsWith(pnames,pat);
    
    nTFindex = find(nTF == 1);
    pTFindex = find(pTF == 1);
    
    if ~isnan(nTFindex) 
        
    X(i) = ninds(nTFindex);
    Y(i) = pinds(pTFindex);
    
    end 
    
end

G = digraph(X,Y);
adj = adjacency(G);
[reactants,products] = find(adj==1);
y_totals = [];
P_totals = {};

for train = 1:n_trains
%     fprintf('train %i \n', train)
    for flash = 1:n_flashes
        P_totallight = eye(size(adj));
        P_totaldark = eye(size(adj));
        P_total = P_totallight*P_totaldark;
        fprintf('flash %i \n', flash)
        k(mult1) = mult1Val;
        k(mult2) = mult2Val;    
        k(n1idx) = n1;
        %flash_duration = 50*10^-6;
        t_lims = [0, flash_duration];
        
        jd30_index = find(strcmp(knames, 'jd30') == 1);
        rqd_index = find(strcmp(knames, 'rqd') == 1);
        rqr_index = find(strcmp(knames, 'rqr') == 1);
        oqr_index = find(strcmp(knames, 'oqr') == 1);
        delta32_index = find(strcmp(knames, 'delta32') == 1);
        %k(rqr_index) = 0.001;
        %k(oqr_index) =  0.001;
        %k(rqd_index) =  0.001;
        %k(delta32_index) =  0.001;
        
        Sol = ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t_lims,yinitial);
        n_timepoints = 500;

        t_vector = linspace(t_lims(1),t_lims(2),n_timepoints+1);
        y = deval(Sol,t_vector);
        n_species = length(species2);
        y_totallight = zeros(n_species, n_timepoints+1);
        y_totallight(:,1) = yinitial(PS2);
        y_totallight = y_totallight';
        y_tmp = yinitial(PS2)';
        deltaT = t_vector(2) - t_vector(1);

        % adjust for PQ and PQH2 pools

        PQ = y(find(strcmp(species,'PQ')==1),:);
        PQH2 = y(find(strcmp(species,'PQH2')==1),:);
        Hs = y(find(strcmp(species,'Hs')==1),:);
        Hl = y(find(strcmp(species,'Hl')==1),:);
        H20_Substrate = y(find(strcmp(species,'H20_substrate')==1),:);

        jd30_index = find(strcmp(knames, 'jd30') == 1);
        rqd_index = find(strcmp(knames, 'rqd') == 1);
        rqr_index = find(strcmp(knames, 'rqr') == 1);
        oqr_index = find(strcmp(knames, 'oqr') == 1);
        delta32_index = find(strcmp(knames, 'delta32') == 1);
        
        rqr = (k(rqr_index)*PQH2);
        oqr = (k(oqr_index)*PQ);
        %rqd = (k(rqd_index)*Hs);
        jd30 = (k(jd30_index)*H20_Substrate);
        delta32 = (k(delta32_index)*Hl);
        y_totallight(1,:) = y_tmp;
        y_totals(flash,:) = y_tmp;
        adj = zeros(length(find(PS2)));
        k2 = k;
        for ii = 2:n_timepoints+1
            k2(rqr_index) = rqr(ii);
            k2(oqr_index) = oqr(ii);
            %k2(rqd_index) = rqd(ii);
            k2(jd30_index) = jd30(ii);
            k2(delta32_index) = delta32(ii);
            for i = 1:length(adj)
                Xinds = find(X == i);
                kconstind = kconst(Xinds);
                kconstval = k2(kconstind);
                ksum = sum(kconstval);
                adj(i,i) = exp(-(ksum*deltaT));
            end
            for counter = 1:length(reactants)
                i = reactants(counter);
                j = products(counter);
                Xinds = find(X == i);
                kconstind = kconst(Xinds);
                kconstvals = k2(kconstind);
                ksum = sum(kconstvals);
                kind = find(X == i & Y == j);
                kconstval = k2(kconst((kind)));
                adj(i,j) = (1 - adj(i,i))*(kconstval/(ksum));
            end
            y_tmp = y_tmp*adj;
            y_totallight(ii,:) = y_tmp;
            P_totallight = P_totallight*adj;
        end
        subplot(2,2,1)
        plot(t_vector, y_totallight')
        title('Markov')
        subplot(2,2,2)
        plot(Sol.x, Sol.y(find(PS2),:))
        title('ODEs')
        [Amarkov,Bmarkov] = maxk(y_totallight(end,:),4);
        [Aode,Bode] = maxk(Sol.y(PS2,end),4);

        diff = Amarkov-Aode';
        
%         for i = 1:length(diff)
%             fprintf('%d, ', diff(i) )
%         end
%         fprintf('\n')

        [sorted_species,sorted_idcs] = sort(species2');
        

        yPS2 = y_totallight(:,sorted_idcs)';
        
        S0_sorted = find(contains(sorted_species,'S0'));
        S1_sorted = find(contains(sorted_species,'S1'));
        S2_sorted = find(contains(sorted_species,'S2'));
        S3_sorted = find(contains(sorted_species,'S3'));
        
%         figure;
%         subplot(2,2,1)
%         plot(t_vector, yPS2(S0_sorted,:))
%         legend(sorted_species(S0_sorted))
%       
%         subplot(2,2,2)
%         plot(t_vector, yPS2(S1_sorted,:))
%         legend(sorted_species(S1_sorted))
%       
%         subplot(2,2,3)
%         plot(t_vector, yPS2(S2_sorted,:))
%         legend(sorted_species(S2_sorted))
%      
%         subplot(2,2,4)
%         plot(t_vector, yPS2(S3_sorted,:))
%         legend(sorted_species(S3_sorted))
        
        yPS2 = Sol.y(find(PS2),:);
        yPS2 = yPS2(sorted_idcs,:);
        
        S0_sorted = find(contains(sorted_species,'S0'));
        S1_sorted = find(contains(sorted_species,'S1'));
        S2_sorted = find(contains(sorted_species,'S2'));
        S3_sorted = find(contains(sorted_species,'S3'));
        
%         figure;
%         subplot(2,2,1)
%         plot(Sol.x, yPS2(S0_sorted,:))
%         legend(sorted_species(S0_sorted))
%         subplot(2,2,2)
%         plot(Sol.x, yPS2(S1_sorted,:))
%         legend(sorted_species(S1_sorted))
%         subplot(2,2,3)
%         plot(Sol.x, yPS2(S2_sorted,:))
%         legend(sorted_species(S2_sorted))
%         subplot(2,2,4)
%         plot(Sol.x, yPS2(S3_sorted,:))
%         legend(sorted_species(S3_sorted))
        
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
        n_timepoints = flash_interval*5000;

%         Soly = ode3(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),linspace(0,flash_interval, n_timepoints),yinitial);
        Sol = ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),t_lims,yinitial);
%         save(['tmp_files\species_concs_vars'], 'k','kconst','rate_inds','S','Rknames','species','t_lims','yinitial')

        subplot(2,2,4)
        plot(Sol.x, Sol.y(PS2,:))
        title('ODEs')
        t_vector = linspace(t_lims(1),t_lims(2),n_timepoints+1);
        y = deval(Sol,t_vector);
        
        n_species = length(species2);
        y_totaldark = zeros(n_species, n_timepoints+1);
        y_totaldark(:,1) = yinitial(PS2);
        y_totaldark = y_totaldark';
        y_tmp = yinitial(PS2)';
        deltaT = t_vector(2) - t_vector(1);

        % adjust for PQ and PQH2 pools

        PQ = y(find(strcmp(species,'PQ')==1),:);
        PQH2 = y(find(strcmp(species,'PQH2')==1),:);
        Hs = y(find(strcmp(species,'Hs')==1),:);
        Hl = y(find(strcmp(species,'Hl')==1),:);
        H20_Substrate = y(find(strcmp(species,'H20_substrate')==1),:);

        jd30_index = find(strcmp(knames, 'jd30') == 1);
        rqd_index = find(strcmp(knames, 'rqd') == 1);
        rqr_index = find(strcmp(knames, 'rqr') == 1);
        oqr_index = find(strcmp(knames, 'oqr') == 1);
        delta32_index = find(strcmp(knames, 'delta32') == 1);

        rqr = (k(rqr_index)*PQH2);
        oqr = (k(oqr_index)*PQ);
        %rqd = (k(rqd_index)*Hs);
        jd30 = (k(jd30_index)*H20_Substrate);
        delta32 = (k(delta32_index)*Hl);
        y_totaldark(1,:) = y_tmp;
        adj = zeros(length(find(PS2)));
        k2 = k;
        for ii = 2:n_timepoints+1
            k2(rqr_index) = rqr(ii);
            k2(oqr_index) = oqr(ii);
            %k2(rqd_index) = rqd(ii);
            k2(jd30_index) = jd30(ii);
            k2(delta32_index) = delta32(ii);
            for i = 1:length(adj)
                Xinds = find(X == i);
                kconstind = kconst(Xinds);
                kconstval = k2(kconstind);
                ksum = sum(kconstval);
                adj(i,i) = exp(-(ksum*deltaT));
            end
            for counter = 1:length(reactants)
                i = reactants(counter);
                j = products(counter);
                Xinds = find(X == i);
                kconstind = kconst(Xinds);
                kconstvals = k2(kconstind);
                ksum = sum(kconstvals);
                kind = find(X == i & Y == j);
                kconstval = k2(kconst((kind)));
                if ksum~=0
                    adj(i,j) = (1 - adj(i,i))*(kconstval/(ksum));
                    if adj(i,j) > 1
                        foo = 1;
                    end
                end
            end
            y_tmp = y_tmp*adj;
            y_totaldark(ii,:) = y_tmp;
            P_totaldark = P_totaldark*adj;
        end
        P_total = P_totallight*P_totaldark;
        P_totals{flash} = P_total;
        subplot(2,2,3)
        plot(t_vector, y_totaldark')
        title('Markov')

        y_final = y_totals(flash,:)*P_total;
        y_diff = y_final - y_tmp;
        
        pause(eps)
        
        yinitial = Sol.y(:,end);
        ts{end+1} = ts{end}(end) + Sol.x;
        ys{end+1} = Sol.y;
        counter = counter+1;
    end
        
%     k(mult1) = 0;
%     k(mult2) = 0;       
%     k(n1idx) = 0;
%     tlims = [0, train_interval];
%     Sol = ode15s(@(t,y) PS2ODES(t,y,k(kconst),k,rate_inds,S,Rknames,species),tlims,yinitial);
%     n_timepoints = train_interval*1000;
%     t_vector = linspace(tlims(1),tlims(2),n_timepoints+1);
%     y = deval(Sol,t_vector);
%     n_species = length(species2);
%     y_totallight = zeros(n_species, n_timepoints+1);
%     y_totallight(:,1) = yinitialmat;
%     y_totallight = y_totallight';
%     deltaT = t_vector(2) - t_vector(1);
% 
%     % adjust for PQ and PQH2 pools
% 
%     PQ = y(find(strcmp(species,'PQ')==1),:);
%     PQH2 = y(find(strcmp(species,'PQH2')==1),:);
%     Hs = y(find(strcmp(species,'Hs')==1),:);
%     Hl = y(find(strcmp(species,'Hl')==1),:);
%     H20_Substrate = y(find(strcmp(species,'H20_substrate')==1),:);
% 
%     jd30_index = find(strcmp(knames, 'jd30') == 1);
%     rqd_index = find(strcmp(knames, 'rqd') == 1);
%     rqr_index = find(strcmp(knames, 'rqr') == 1);
%     oqr_index = find(strcmp(knames, 'oqr') == 1);
%     delta32_index = find(strcmp(knames, 'delta32') == 1);
%     k2 = k;
%     rqr = (k(rqr_index)*PQH2);
%     oqr = (k(oqr_index)*PQ);
%     %rqd = (k(rqd_index)*Hs);
%     jd30 = (k(jd30_index)*H20_Substrate);
%     delta32 = (k(delta32_index)*Hl);
%     y_totallight(1,:) = y_tmp;
% 
%     for ii = 2:n_timepoints+1
%         k2(rqr_index) = rqr(ii);
%         k2(oqr_index) = oqr(ii);
%         %k2(rqd_index) = rqd(ii);
%         k2(jd30_index) = jd30(ii);
%         k2(delta32_index) = delta32(ii);
%         for reactant = 1:length(adj)
%             Xinds = find(X == i);
%             kconstind = kconst(Xinds);
%             kconstval = k2(kconstind);
%             ksum = sum(kconstval);
%             adj(i,i) = exp(-(ksum*deltaT));
%         end
%         for counter = 1:length(reactants)
%             reactant = reactants(counter);
%             product = products(counter);
%             Xinds = find(X == reactant);
%             kconstind = kconst(Xinds);
%             kconstvals = k2(kconstind);
%             ksum = sum(kconstvals);
%             kind = find(X == reactant & Y == product);
%             kconstval = k2(kconst((kind)));
%             adj(reactant,product) = (1 - adj(reactant,reactant))*(kconstval/(ksum));
%         end
%         y_tmp = y_tmp*adj;
%         y_totallight(ii,:) = y_tmp;
%         %P_totallight = P_totallight*adj;
%     end   
%     yinitial = Sol.y(:,end);
%     ts{end+1} = ts{end}(end) + Sol.x;
%     ys{end+1} = Sol.y;

end
end
