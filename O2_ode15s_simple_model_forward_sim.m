function [O2_sim, O2_exp] = O2_ode15s_simple_model_forward_sim(analysis_name,result_name)



file1 = [analysis_name,'/O2_exp.csv'];
C = readtable(file1);
raw_data = C.O2;
average_data = sum(raw_data)/length(raw_data); 
O2 = raw_data/average_data; 
O2_exp = O2;


file1 = [analysis_name,'/experimental_parameters.xls'];
tbl1 = readtable(file1);
n_flashes = tbl1.n_flashes;
flash_duration = tbl1.flash_duration;
flash_interval = tbl1.flash_interval;
train_interval = tbl1.train_interval;
n_trains = tbl1.n_trains;
% 
% n_flashes = 50;
% flash_duration = 5e-5;
% % intervals = [1];
% intervals = 1;
% train_interval = .0001;

file1 = [analysis_name,'/LaiskConstants.xls'];
tablek = readtable(file1);
%tablek.base_val(indepkres) = resvalsk;
%tablek.independent(indepkres) = 1;
indepk = find(tablek.independent);
lbk = tablek.lb(indepk);
ubk = tablek.ub(indepk);
knames = tablek.name;
k_std = tablek.base_val;

file2 = [analysis_name,'/LaiskY.xls'];
tabley = readtable(file2);
indepy = find(tabley.independent);


%tabley.base_val(indepyres) = resvalsy;
%tabley.independent(indepyres) = 1;
indepy = find(tabley.independent);
y_std = tabley.base_val;



lby = tabley.lb(indepy);
uby = tabley.ub(indepy);
% yr = lby + (uby-lby).*rand(length(indepy),1);
% kr = lbk + (ubk-lbk).*rand(length(indepk),1);
% yr = yr+ yr.*rand(length(yr),1)*.5 - yr.*rand(length(yr),1)*.5;
kr = k_std(indepk);
yr = y_std(indepy);
% kr = kr+ kr.*rand(length(kr),1)*.5 - kr.*rand(length(kr),1)*.5;

indep_ps2 = find(contains(tabley.name(indepy), 'Y'));
Ay = zeros(1,length(yr));
Ay(indep_ps2) = 1;
by = 1;

Ak = zeros(1,length(indepk));
Aeq = [Ay,Ak];
beq = 1;
x0 = [yr;kr];

lb = [reshape(lby,[],1); reshape(lbk,[],1)];
ub = [reshape(uby,[],1); reshape(ubk,[],1)];

Ynames = tabley.name;
file3 = [analysis_name,'/LaiskReactions.xlsx'];
[~,Rknames] = xlsread(file3);

PFD = find(strcmp(knames, 'PFD')); 
a2 = find(strcmp(knames, 'a2'));
b1d = find(strcmp(knames, 'b1d'));
b2d = find(strcmp(knames, 'b2d'));
Chl = find(strcmp(knames, 'Chl')); 
CytfT = find(strcmp(knames, 'CytfT')); 
FDT = find(strcmp(knames, 'FDT')); 
jd = find(strcmp(knames, 'jd')); 
kb6f = find(strcmp(knames, 'kb6f')); 
kcytf = find(strcmp(knames, 'kcytf')); 
kf = find(strcmp(knames, 'kf')); 
kfd = find(strcmp(knames, 'kfd')); 
kfx = find(strcmp(knames, 'kfx')); 
kn = find(strcmp(knames, 'kn')); 
kp = find(strcmp(knames, 'kp'));
kpc = find(strcmp(knames, 'kpc'));
kr = find(strcmp(knames, 'kr')); 
kE1 = find(strcmp(knames, 'kE1'));
kEb6f = find(strcmp(knames, 'kEb6f'));
kEcytf = find(strcmp(knames,'kEcytf'));
kEfx = find(strcmp(knames,'kEfx'));
kEpc = find(strcmp(knames,'kEpc'));
Labs = find(strcmp(knames,'Labs'));
oqd = find(strcmp(knames,'oqd'));
oqr = find(strcmp(knames,'oqr'));
PCT = find(strcmp(knames,'PCT'));
PQT = find(strcmp(knames,'PQT'));
PSU1 = find(strcmp(knames,'PSU1'));
PSU2 = find(strcmp(knames,'PSU2'));
rqd = find(strcmp(knames,'rqd'));
rqr = find(strcmp(knames,'rqr'));
b1r = find(strcmp(knames,'b1r'));
kq = find(strcmp(knames,'kq')); 
P700T = find(strcmp(knames, 'P700T'));
FXT = find(strcmp(knames, 'FXT')); 
 
YoPoAo = find(strcmp(Ynames,'YoPoAo')); 
YoPoAoBoo = find(strcmp(Ynames,'YoPoAoBoo')); 
YoPrAo = find(strcmp(Ynames,'YoPrAo')); 
YoPoAr = find(strcmp(Ynames,'YoPoAr')); 
YoPrAoBoo = find(strcmp(Ynames,'YoPrAoBoo')); 
YoPoArBoo = find(strcmp(Ynames,'YoPoArBoo')); 
YoPoAoBro = find(strcmp(Ynames,'YoPoAoBro')); 
YrPrAo = find(strcmp(Ynames,'YrPrAo')); 
YoPrAr = find(strcmp(Ynames,'YoPrAr')); 
YrPrAoBoo = find(strcmp(Ynames,'YrPrAoBoo')); 
YoPrArBoo = find(strcmp(Ynames,'YoPrArBoo')); 
YoPrAoBro = find(strcmp(Ynames,'YoPrAoBro')); 
YoPoArBro = find(strcmp(Ynames,'YoPoArBro')); 
YoPoAoBrr = find(strcmp(Ynames,'YoPoAoBrr')); 
YrPrAr = find(strcmp(Ynames,'YrPrAr')); 
YrPrArBoo = find(strcmp(Ynames,'YrPrArBoo')); 
YoPrArBro = find(strcmp(Ynames,'YoPrArBro')); 
YoPrAoBrr = find(strcmp(Ynames,'YoPrAoBrr')); 
YrPrAoBro = find(strcmp(Ynames,'YrPrAoBro')); 
YoPoArBrr = find(strcmp(Ynames,'YoPoArBrr')); 
YrPrArBro = find(strcmp(Ynames,'YrPrArBro')); 
YrPrAoBrr = find(strcmp(Ynames,'YrPrAoBrr')); 
YoPrArBrr = find(strcmp(Ynames,'YoPrArBrr')); 
YrPrArBrr = find(strcmp(Ynames,'YrPrArBrr')); 
PQH2 = find(strcmp(Ynames,'PQH2')); 
PQ = find(strcmp(Ynames,'PQ')); 
Cytfr = find(strcmp(Ynames,'Cytfr')); 
PCr = find(strcmp(Ynames,'PCr')); 
P700r = find(strcmp(Ynames,'P700r')); 
FXr = find(strcmp(Ynames,'FXr')); 
FDr = find(strcmp(Ynames,'FDr')); 
PCo = find(strcmp(Ynames, 'PCo'));
P700o = find(strcmp(Ynames, 'P700o'));
FXo = find(strcmp(Ynames, 'FXo'));
FDo = find(strcmp(Ynames, 'FDo'));
Cytfo = find(strcmp(Ynames, 'Cytfo')); 
O2 = find(strcmp(Ynames, 'O2')); 
Fl = find(strcmp(Ynames, 'Fl')); 
 
% PS1T = k(a2)*k(Chl)/k(PSU2); 
% PS2T = (1-k(a2))*(k(Chl)/k(PSU1));
% PS2TR = (1-k(a2))*(k(Chl)/k(PSU1));

% PS1T = 1; 
% PS2T = 1;
% n1 = k(PFD)*k(Labs)*(1-k(a2))/PS1T;
% n2 = k(PFD)*k(Labs)*k(a2)/PS2T; 
% y0(P700r) = PS1T/PS2T;
% y0(FXo) = PS1T/PS2T;



% 
% k(oqr) = k(oqr);
% k(rqr) = k(rqr);
% k(kb6f) = k(kb6f);
% k(kcytf) = k(kcytf);
% k(kpc) = k(kpc);
% k(kfx) = k(kfx);
% k(P700T) = PS1T/PS2T; 
% k(FXT) = PS1T/PS2T; 
% k(kfd) = k(kfd);
mult1 = find(strcmp(knames,'n2*kp/(1+kp+kn+kr)'));
mult2 = find(strcmp(knames,'n2*kp/(1+kp+kn)')); 
Div1 = find(strcmp(knames,'kpc/kEpc'));
Div2 = find(strcmp(knames,'kcytf/kEcytf'));
Div3 = find(strcmp(knames,'kfx/kEfx'));
Div4 = find(strcmp(knames,'kb6f/kEb6f'));
n1idx = find(strcmp(knames,'n1'));

% mult1Val = n2*k(kp)/(1+k(kp)+k(kn)+k(kr));
% mult2Val = n2*k(kp)/(1+k(kp)+k(kn));
% Div1Val = k(kpc)/k(kEpc);
% Div2Val = k(kcytf)/k(kEcytf);
% Div3Val = k(kfx)/k(kEfx);
% Div4Val = k(kb6f)/k(kEb6f);
% 
% k(mult1) = mult1Val;
% k(mult2) = mult2Val;
% k(Div1) = Div1Val;
% k(Div2) = Div2Val;
% k(Div3) = Div3Val; 
% k(Div4) = Div4Val;
% k(n1idx) = n1;

% idcs = struct;
kidcs.PFD = PFD;
kidcs.Labs = Labs;
kidcs.a2 = a2;
kidcs.PSU2 = PSU2;
kidcs.Chl = Chl;
kidcs.PSU1 = PSU1;
kidcs.kp = kp;
kidcs.kn = kn;
kidcs.kr = kr;
kidcs.Div1 = Div1;
kidcs.Div2 = Div2;
kidcs.Div3 = Div3;
kidcs.Div4 = Div4;
kidcs.mult1 = mult1;
kidcs.mult2 = mult2;
kidcs.n1idx = n1idx;
kidcs.kpc = kpc;
kidcs.kEpc = kEpc;
kidcs.kcytf = kcytf;
kidcs.kEcytf = kEcytf;
kidcs.kfx = kfx;
kidcs.kEfx = kEfx;
kidcs.kb6f = kb6f;
kidcs.kEb6f = kEb6f;
kidcs.kf = kf;


[species,S,rate_inds] = Laisk_read_excel_model(analysis_name);
species_idcs = zeros(length(species),1);
% yinitial = zeros(length(y0),1);
for i = 1:length(Ynames)
    index = find(strcmp(species,Ynames(i)));
%     yinitial(index) = y0(i);
    species_idcs(index) = i;
end
PSIidcs = zeros(2,1);
pat = "S";
PS2 = startsWith(species,pat);
species2 = species(PS2);

% PSIidcs(1) = find(strcmp(species,'P700r'));%- Commented out for the simpler model
% up to PQ
% PSIidcs(2) = find(strcmp(species,'FXr')); %- Commented out for the simpler model
% up to PQ

[kconst] = LaiskKconstantsReadTable(analysis_name);


kn = find(strcmp(knames,'kn'));
kp = find(strcmp(knames,'kp'));
kr = find(strcmp(knames,'kr')); 
kq = find(strcmp(knames,'kq')); 
Fluorescence_k_idcs = [kn;kp;kr;kq];

yopoax = find(contains(species,'YoPoA'));
yoprao = find(contains(species,'YoPrAo'));
yoprar = find(contains(species,'YoPrAr'));
yrprao = find(contains(species,'YrPrAo'));
yrprar = find(contains(species,'YrPrAr'));
Fluorescence_y_inds = {yopoax;yoprao;yoprar;yrprao;yrprar};

load(['results/', analysis_name, '/' result_name],'O2_sim_opt','O2_exp')
xopt = x0;
n_flashes = 40;
%disp(resnames)

[O2_sim] = calc_O2_simple_model_ode15s(xopt, n_trains, n_flashes, ...
                    flash_duration, flash_interval, train_interval, ... 
                    Fluorescence_k_idcs, Fluorescence_y_inds, kidcs, ... 
                    tablek, tabley, kconst, rate_inds, ... 
                    S, species, species_idcs, Rknames);
                
[ts,ys,FvFm] = calc_species_concs_simple_model_ode15s(xopt,... Set of parameters. This only includes the independent variables as described by the third column in Y and Constants files
                    n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... Experimental parameters
                    Fluorescence_k_idcs, Fluorescence_y_inds,... indeces used to calculate fluorescence
                    kidcs, PSIidcs, ... all indices needed in to calculate FvFm and prepare the variables
                    tablek, tabley,... information on the k and y variables
                    kconst, rate_inds, S, species, knames, species_idcs, Rknames); % model specific variables

O2_sim = O2_sim/mean(O2_sim);
O2_sim_opt = O2_sim_opt/mean(O2_sim_opt);
figure
plot(1:length(O2_exp), O2_exp,'-o', 'color','black','DisplayName','O2_experimental');
hold on;
plot(1:length(O2_sim), O2_sim,'-o', 'color','red','DisplayName','O2_experimental');
[RMSE] = sqrt(mean((O2_exp(1:n_flashes)-O2_sim(1:n_flashes)).^2))
foo = 1;
f = plot_pq_redox_state(species, ys, ts);
% pq = strcmp(species,'PQ');
% pqh2 = strcmp(species,'PQH2');
% boo = find(contains(species,'Boo'));
% bro = find(contains(species,'Bro'));
% brr = find(contains(species,'Brr'));

% t = [];
% PQ = [];
% PQH2 = [];
% BOO = [];
% BRO = [];
% BRR = [];
% figure;
% for itime = 2:length(ys)
%     %plot(ts{itime},sum(ys{itime}(boo,:)))
%     PQ = [PQ, ys{itime}(pq,:)];
%     PQH2 = [PQH2, ys{itime}(pqh2,:)];
%     BOO = [BOO, sum(ys{itime}(boo,:))];
%     BRO = [BRO, sum(ys{itime}(bro,:))];
%     BRR = [BRR, sum(ys{itime}(brr,:))];
%     t = [t, ts{itime}];
% end
% hold on
% plot(t, PQH2./(PQ+PQH2))
% z = PQH2./(PQ+PQH2);

plot_S_states(species, ys, ts)
plot_Qb_site(species, ys, ts)
plot_QaQb_site(species, ys, ts)
plot_QaQbBxx_site(species, ys, ts)
plot_S3yrprarbrr_s1yrpraobro(species, ys, ts)
plot_S3_S1_states(species, ys, ts)


rs = calc_reaction_rates(ys, tablek.base_val(kconst), rate_inds);
plot_reaction_rates([156, 236, 259, 269], rs, ts)

end





