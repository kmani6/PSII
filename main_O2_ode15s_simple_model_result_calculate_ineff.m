function [O2_sim, O2_exp] = main_O2_ode15s_simple_model_result_calculate_ineff(analysis_name,result_name)


load(['results/', analysis_name, '/' result_name],'tablek','knames','tabley')
indepkres = find(tablek.independent);
resnamesk = knames(indepkres);
resvalsk = tablek.base_val(indepkres);
indepyres = find(tabley.independent);
ynames = tabley.name;
resnamesy = ynames(indepyres);
resvalsy = tabley.base_val(indepyres);
resnames = [resnamesy;resnamesk];

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
tablek.base_val(indepkres) = resvalsk;
tablek.independent(indepkres) = 1;
indepk = find(tablek.independent);
lbk = tablek.lb(indepk);
ubk = tablek.ub(indepk);
knames = tablek.name;
k_std = tablek.base_val;

file2 = [analysis_name,'/LaiskY.xls'];
tabley = readtable(file2);
tabley.base_val(indepyres) = resvalsy;
tabley.independent(indepyres) = 1;
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
pq = strcmp(species,'PQ');
pqh2 = strcmp(species,'PQH2');
boo = find(contains(species,'Boo'));
bro = find(contains(species,'Bro'));
brr = find(contains(species,'Brr'));

t = [];
PQ = [];
PQH2 = [];
BOO = [];
BRO = [];
BRR = [];

for itime = 2:length(ys)
    plot(ts{itime},sum(ys{itime}(boo,:)))
    PQ = [PQ, ys{itime}(pq,:)];
    PQH2 = [PQH2, ys{itime}(pqh2,:)];
    BOO = [BOO, sum(ys{itime}(boo,:))];
    BRO = [BRO, sum(ys{itime}(bro,:))];
    BRR = [BRR, sum(ys{itime}(brr,:))];
    t = [t, ts{itime}];
end
hold on
plot(t, PQH2./(PQ+PQH2))
z = PQH2./(PQ+PQH2);
plot_Qb_site(species, ys, ts)
plot_QaQb_site(species, ys, ts)
plot_QaQbBxx_site(species, ys, ts)

[ts,ys,FvFm, y_totaltrain,P_totals] = calc_species_concs_inefficiencies_simple_model_ode15s(xopt,... Set of parameters. This only includes the independent variables as described by the third column in Y and Constants files
    n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... Experimental parameters
    Fluorescence_k_idcs, Fluorescence_y_inds,... indeces used to calculate fluorescence
    kidcs, PSIidcs, ... all indices needed in to calculate FvFm and prepare the variables
    tablek, tabley,... information on the k and y variables
    kconst, rate_inds, S, species, knames, species_idcs, Rknames); % model specific variables
  
% save(['results/', '/Optimal_parameters_' datestr(now, 30)],...
%     'xopt', 'y_totaltrain','P_totals','ts','ys')
 
% inefficiences_name = 'Optimal_parameters_20210120T195957';
% load(['results/','/', inefficiences_name, '.mat'])

VZAD = [0.0021103166939887156
0.19987592358114067
1.8925013683226881
1.5207135203885611
0.7393772090279851
0.5013496776778973
1.2407657967953938
1.3458660048777429
1.018635194973857
0.7648492349497104
1.0263260600576327
1.1780580203532347
1.0836203087049034
0.9202418654170555
0.9830648162218205
1.080265275155017
1.0746102120504908
0.9922726762652906
0.9902106792422243
1.0358761696341854
1.051796730888972
1.017935323582101
1.0034907600458334
1.0195519948234273
1.033776600134401
1.0229088006047022
1.0115529945805435
1.0146324419458974
1.0227318655346331
1.020609212805405
1.0142097133921253
1.0130333281370976
1.0164254026590704
1.0166717448061646
1.0136060807572833
1.0117220784872676
1.0125391492336
1.0129071695725387
1.011498795379536
1.0099430001317873
1.009641070545703
1.0096216862093708
1.0088583026953473
1.0077225182545606
1.0070461714557513
1.0066899846742048
1.0061030999930642
1.0052424571268275
1.0044950099539478
1.003947695664409];

%Determine Ineffiency Parameters

%gets list of PSII species for each of the redox states
IndexC = strfind(species2,'AoBoo');
ZeroEIdcs = (not(cellfun('isempty',IndexC)));
ZeroEIdcsAoBoo = find(not(cellfun('isempty',IndexC)));

IndexC = strfind(species2,'ArBoo');
OneEIdcsArBoo = (not(cellfun('isempty',IndexC)));
OneEIdcsArBoo1 = find(not(cellfun('isempty',IndexC)));
IndexC = strfind(species2,'AoBro');
OneEIdcsBro = (not(cellfun('isempty',IndexC)));
OneEIdcsBro1 = find(not(cellfun('isempty',IndexC)));

IndexC = strfind(species2,'ArBro');
TwoEIdcsArBro = (not(cellfun('isempty',IndexC)));
TwoEIdcsArBro1 = find(not(cellfun('isempty',IndexC)));
IndexC = strfind(species2,'AoBrr');
TwoEIdcsAoBrr = (not(cellfun('isempty',IndexC)));
TwoEIdcsAoBrr1 = find(not(cellfun('isempty',IndexC)));

IndexC = strfind(species2,'ArBrr');
ThreeEIdcs = (not(cellfun('isempty',IndexC)));
ThreeEIdcs1 = find(not(cellfun('isempty',IndexC)));

species2length = ones(1,96);
RemainingIdcs = find(species2length ~= ZeroEIdcs & species2length ~= OneEIdcsArBoo & species2length ~= OneEIdcsBro & species2length ~= TwoEIdcsArBro & species2length ~= TwoEIdcsAoBrr & species2length ~= ThreeEIdcs);
Newspecies = {1:96};
Newspecies(1:96) = {'foo'};
Newspecies(RemainingIdcs) = species2(RemainingIdcs);

IndexC = strfind(Newspecies,'Ao');
ZeroEIdcsAo = find(not(cellfun('isempty',IndexC)));
IndexC = strfind(Newspecies,'Ar');
OneEIdcsAr = find(not(cellfun('isempty',IndexC)));

ZeroEIdcs = [ZeroEIdcsAo,ZeroEIdcsAoBoo];
ZeroEIdc = zeros(1,96);
ZeroEIdc(ZeroEIdcs) = 1;
OneEIdcs = [OneEIdcsAr,OneEIdcsBro1,OneEIdcsArBoo1];
OneEIdc = zeros(1,96);
OneEIdc(OneEIdcs) = 1;
TwoEIdcs = [TwoEIdcsArBro1,TwoEIdcsAoBrr1];
TwoEIdc = zeros(1,96);
TwoEIdc(TwoEIdcs) = 1;
ThreeEIdcs = ThreeEIdcs1;
ThreeEIdc = zeros(1,96);
ThreeEIdc(ThreeEIdcs) = 1;

%gets list of PSII species for each of the S-states
IndexC = strfind(species2,'S0');
S0Idcs = find(not(cellfun('isempty',IndexC)));
S0Idc = zeros(1,96);
S0Idc(S0Idcs) = 1;
IndexC = strfind(species2,'S1');
S1Idcs = find(not(cellfun('isempty',IndexC)));
S1Idc = zeros(1,96);
S1Idc(S1Idcs) = 1;
IndexC = strfind(species2,'S2');
S2Idcs = find(not(cellfun('isempty',IndexC)));
S2Idc = zeros(1,96);
S2Idc(S2Idcs) = 1;
IndexC = strfind(species2,'S3');
S3Idcs = find(not(cellfun('isempty',IndexC)));
S3Idc = zeros(1,96);
S3Idc(S3Idcs) = 1;

%gets list of PSII species for each of the electron bins with respect to
%State
S0IdcsE0 = find(S0Idc == 1 & ZeroEIdc == 1);
S1IdcsE0 = find(S1Idc == 1 & ZeroEIdc == 1);
S2IdcsE0 = find(S2Idc == 1 & ZeroEIdc == 1);
S3IdcsE0 = find(S3Idc == 1 & ZeroEIdc == 1);

S0IdcsE1 = find(S0Idc == 1 & OneEIdc == 1);
S1IdcsE1 = find(S1Idc == 1 & OneEIdc == 1);
S2IdcsE1 = find(S2Idc == 1 & OneEIdc == 1);
S3IdcsE1 = find(S3Idc == 1 & OneEIdc == 1);

S0IdcsE2 = find(S0Idc == 1 & TwoEIdc == 1);
S1IdcsE2 = find(S1Idc == 1 & TwoEIdc == 1);
S2IdcsE2 = find(S2Idc == 1 & TwoEIdc == 1);
S3IdcsE2 = find(S3Idc == 1 & TwoEIdc == 1);

S0IdcsE3 = find(S0Idc == 1 & ThreeEIdc == 1);
S1IdcsE3 = find(S1Idc == 1 & ThreeEIdc == 1);
S2IdcsE3 = find(S2Idc == 1 & ThreeEIdc == 1);
S3IdcsE3 = find(S3Idc == 1 & ThreeEIdc == 1);

%get list of PSII species for each hole number

ZeroHoleIdcs1 = [ThreeEIdcs1, OneEIdcsAr]; 
ZeroHoleIdc = zeros(1,96);
ZeroHoleIdc(ZeroHoleIdcs1) = 1; 
OneHoleIdcs1 = [TwoEIdcsArBro1,ZeroEIdcsAo,TwoEIdcsAoBrr1];
OneHoleIdc = zeros(1,96);
OneHoleIdc(OneHoleIdcs1) = 1; 
TwoHoleIdcs1 = [OneEIdcsArBoo1, OneEIdcsBro1];
TwoHoleIdc = zeros(1,96);
TwoHoleIdc(TwoHoleIdcs1) = 1; 
ThreeHoleIdcs1 = ZeroEIdcsAoBoo;
ThreeHoleIdc = zeros(1,96);
ThreeHoleIdc(ThreeHoleIdcs1) = 1; 

S0IdcsH0 = find(S0Idc == 1 & ZeroHoleIdc == 1);
S1IdcsH0 = find(S1Idc == 1 & ZeroHoleIdc == 1);
S2IdcsH0 = find(S2Idc == 1 & ZeroHoleIdc == 1);
S3IdcsH0 = find(S3Idc == 1 & ZeroHoleIdc == 1);

S0IdcsH1 = find(S0Idc == 1 & OneHoleIdc == 1);
S1IdcsH1 = find(S1Idc == 1 & OneHoleIdc == 1);
S2IdcsH1 = find(S2Idc == 1 & OneHoleIdc == 1);
S3IdcsH1 = find(S3Idc == 1 & OneHoleIdc == 1);

S0IdcsH2 = find(S0Idc == 1 & TwoHoleIdc == 1);
S1IdcsH2 = find(S1Idc == 1 & TwoHoleIdc == 1);
S2IdcsH2 = find(S2Idc == 1 & TwoHoleIdc == 1);
S3IdcsH2 = find(S3Idc == 1 & TwoHoleIdc == 1);

S0IdcsH3 = find(S0Idc == 1 & ThreeHoleIdc == 1);
S1IdcsH3 = find(S1Idc == 1 & ThreeHoleIdc == 1);
S2IdcsH3 = find(S2Idc == 1 & ThreeHoleIdc == 1);
S3IdcsH3 = find(S3Idc == 1 & ThreeHoleIdc == 1);

S0fracs = []; S1fracs = []; S2fracs = []; S3fracs = [];
for flash = 1:n_flashes
S0fracs(end+1) = sum(y_totaltrain(flash,S0Idcs))/sum(y_totaltrain(flash,:));
S1fracs(end+1) = sum(y_totaltrain(flash,S1Idcs))/sum(y_totaltrain(flash,:));
S2fracs(end+1) = sum(y_totaltrain(flash,S2Idcs))/sum(y_totaltrain(flash,:));
S3fracs(end+1) = sum(y_totaltrain(flash,S3Idcs))/sum(y_totaltrain(flash,:));
end
% figure
% plot(1:50,S0fracs,'-o','color','black');

%get alpha parameter
PS0miss = zeros(1,n_flashes);
PS1miss = zeros(1,n_flashes);
PS2miss = zeros(1,n_flashes);
PS3miss = zeros(1,n_flashes);

for k = 1:n_flashes
    P_total = P_totals{k};
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS0miss(k) = PS0miss(k) + y_totaltrain(k,S0Idcs(i))*P_total(S0Idcs(i), S0Idcs(j));
        end
    end
    PS0miss(k) = PS0miss(k)/sum(y_totaltrain(k,S0Idcs));
    for i = 1:length(S1Idcs)
        for j = 1:length(S1Idcs)
            PS1miss(k) = PS1miss(k) + y_totaltrain(k,S1Idcs(i))*P_total(S1Idcs(i), S1Idcs(j));
        end
    end
    PS1miss(k) = PS1miss(k)/sum(y_totaltrain(k,S1Idcs));
    for i = 1:length(S2Idcs)
        for j = 1:length(S2Idcs)
            PS2miss(k) = PS2miss(k) + y_totaltrain(k,S2Idcs(i))*P_total(S2Idcs(i), S2Idcs(j));
        end
    end
    PS2miss(k) = PS2miss(k)/sum(y_totaltrain(k,S2Idcs));
    for i = 1:length(S3Idcs)
        for j = 1:length(S3Idcs)
            PS3miss(k) = PS3miss(k) + y_totaltrain(k,S3Idcs(i))*P_total(S3Idcs(i), S3Idcs(j));
        end
    end
    PS3miss(k) = PS3miss(k)/sum(y_totaltrain(k,S3Idcs));
end

%Get Total Miss Parameter
S0 = S0fracs.*PS0miss;
S1 = S1fracs.*PS1miss;
S2 = S2fracs.*PS2miss;
S3 = S3fracs.*PS3miss;
Totalmiss = S0 + S1 + S2 + S3;
Totalmiss(1) = S0(1) + S1(1);
%Totalmisses(ky) = (Totalmiss(end));          

%Get Miss Parameter Electron Holes
% PlotAlphaElectronHoles(n_flashes,P_totals,y_totaltrain,S0IdcsH0,S1IdcsH0,S2IdcsH0,S3IdcsH0,...
%                                     S0IdcsH1,S1IdcsH1,S2IdcsH1,S3IdcsH1,S0IdcsH2,S1IdcsH2,S2IdcsH2,S3IdcsH2,...
%                                     S0IdcsH3,S1IdcsH3,S2IdcsH3,S3IdcsH3,S0Idcs,S1Idcs,S2Idcs,S3Idcs);
%Get Miss Parameter Electron Bins                            
% PlotAlphaElectronBins(n_flashes,P_totals,y_totaltrain,S0IdcsE0,S1IdcsE0,S2IdcsE0,S3IdcsE0,...
%                                     S0IdcsE1,S1IdcsE1,S2IdcsE1,S3IdcsE1,S0IdcsE2,S1IdcsE2,S2IdcsE2,S3IdcsE2,...
%                                     S0IdcsE3,S1IdcsE3,S2IdcsE3,S3IdcsE3,S0Idcs,S1Idcs,S2Idcs,S3Idcs)                                
%get beta parameter
PS02doublehit = zeros(1,n_flashes);
PS13doublehit = zeros(1,n_flashes);
PS20doublehit = zeros(1,n_flashes);
PS31doublehit = zeros(1,n_flashes);

for k = 1:n_flashes
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS02doublehit(k) = PS02doublehit(k) + y_totaltrain(k,S0Idcs(i))*P_total(S0Idcs(i), S2Idcs(j));
        end
    end
    PS02doublehit(k) = PS02doublehit(k)/sum(y_totaltrain(k,S0Idcs));
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS13doublehit(k) = PS13doublehit(k) + y_totaltrain(k,S1Idcs(i))*P_total(S1Idcs(i), S3Idcs(j));
        end
    end
    PS13doublehit(k) = PS13doublehit(k)/sum(y_totaltrain(k,S1Idcs));
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS20doublehit(k) = PS20doublehit(k) + y_totaltrain(k,S2Idcs(i))*P_total(S2Idcs(i), S0Idcs(j));
        end
    end
    PS20doublehit(k) = PS20doublehit(k)/sum(y_totaltrain(k,S2Idcs));
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS31doublehit(k) = PS31doublehit(k) + y_totaltrain(k,S3Idcs(i))*P_total(S3Idcs(i), S1Idcs(j));
        end
    end
    PS31doublehit(k) = PS31doublehit(k)/sum(y_totaltrain(k,S3Idcs));
end

%Get Total Beta Parameter
S0 = S0fracs.*PS02doublehit;
S1 = S1fracs.*PS13doublehit;
S2 = S2fracs.*PS20doublehit;
S3 = S3fracs.*PS31doublehit;
TotalBeta = S0 + S1 + S2 + S3;
TotalBeta(1) = S0(1) + S1(1);     
%TotalBetas(ky) = (TotalBeta(end));    

%Get Beta Parameter Electron Holes
% PlotBetaElectronHoles(n_flashes,P_totals,y_totaltrain,S0IdcsH0,S1IdcsH0,S2IdcsH0,S3IdcsH0,...
%                                     S0IdcsH1,S1IdcsH1,S2IdcsH1,S3IdcsH1,S0IdcsH2,S1IdcsH2,S2IdcsH2,S3IdcsH2,...
%                                     S0IdcsH3,S1IdcsH3,S2IdcsH3,S3IdcsH3,S0Idcs,S1Idcs,S2Idcs,S3Idcs);
%Get Miss Parameter Electron Bins                            
% PlotBetaElectronBins(n_flashes,P_totals,y_totaltrain,S0IdcsE0,S1IdcsE0,S2IdcsE0,S3IdcsE0,...
%                                     S0IdcsE1,S1IdcsE1,S2IdcsE1,S3IdcsE1,S0IdcsE2,S1IdcsE2,S2IdcsE2,S3IdcsE2,...
%                                     S0IdcsE3,S1IdcsE3,S2IdcsE3,S3IdcsE3,S0Idcs,S1Idcs,S2Idcs,S3Idcs);                                    
%get gamma parameter
PS01singlehit = zeros(1,n_flashes);
PS12singlehit = zeros(1,n_flashes);
PS23singlehit = zeros(1,n_flashes);
PS30singlehit = zeros(1,n_flashes);

for k = 1:n_flashes
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS01singlehit(k) = PS01singlehit(k) + y_totaltrain(k,S0Idcs(i))*P_total(S0Idcs(i), S1Idcs(j));
        end
    end
    PS01singlehit(k) = PS01singlehit(k)/sum(y_totaltrain(k,S0Idcs));
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS12singlehit(k) = PS12singlehit(k) + y_totaltrain(k,S1Idcs(i))*P_total(S1Idcs(i), S2Idcs(j));
        end
    end
    PS12singlehit(k) = PS12singlehit(k)/sum(y_totaltrain(k,S1Idcs));
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS23singlehit(k) = PS23singlehit(k) + y_totaltrain(k,S2Idcs(i))*P_total(S2Idcs(i), S3Idcs(j));
        end
    end
    PS23singlehit(k) = PS23singlehit(k)/sum(y_totaltrain(k,S2Idcs));
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS30singlehit(k) = PS30singlehit(k) + y_totaltrain(k,S3Idcs(i))*P_total(S3Idcs(i), S0Idcs(j));
        end
    end
    PS30singlehit(k) = PS30singlehit(k)/sum(y_totaltrain(k,S3Idcs));
end

%Get Total Gamma Parameter
S0 = S0fracs.*PS01singlehit;
S1 = S1fracs.*PS12singlehit;
S2 = S2fracs.*PS23singlehit;
S3 = S3fracs.*PS30singlehit;
TotalGamma = S0 + S1 + S2 + S3;
TotalGamma(1) = S0(1) + S1(1);

%Get Gamma Parameter Electron Holes
% PlotGammaElectronHoles(n_flashes,P_totals,y_totaltrain,S0IdcsH0,S1IdcsH0,S2IdcsH0,S3IdcsH0,...
%                                     S0IdcsH1,S1IdcsH1,S2IdcsH1,S3IdcsH1,S0IdcsH2,S1IdcsH2,S2IdcsH2,S3IdcsH2,...
%                                     S0IdcsH3,S1IdcsH3,S2IdcsH3,S3IdcsH3,S0Idcs,S1Idcs,S2Idcs,S3Idcs);
%Get Gamma Parameter Electron Bins                            
% PlotGammaElectronBins(n_flashes,P_totals,y_totaltrain,S0IdcsE0,S1IdcsE0,S2IdcsE0,S3IdcsE0,...
%                                     S0IdcsE1,S1IdcsE1,S2IdcsE1,S3IdcsE1,S0IdcsE2,S1IdcsE2,S2IdcsE2,S3IdcsE2,...
%                                     S0IdcsE3,S1IdcsE3,S2IdcsE3,S3IdcsE3,S0Idcs,S1Idcs,S2Idcs,S3Idcs);         
%get delta parameter
PS21backwards = zeros(1,n_flashes);
PS32backwards = zeros(1,n_flashes);

for k = 1:n_flashes
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS21backwards(k) = PS21backwards(k) + y_totaltrain(k,S2Idcs(i))*P_total(S2Idcs(i), S1Idcs(j));
        end
    end
    PS21backwards(k) = PS21backwards(k)/sum(y_totaltrain(k,S2Idcs));
    for i = 1:length(S0Idcs)
        for j = 1:length(S0Idcs)
            PS32backwards(k) = PS32backwards(k) + y_totaltrain(k,S3Idcs(i))*P_total(S3Idcs(i), S2Idcs(j));
        end
    end
    PS32backwards(k) = PS32backwards(k)/sum(y_totaltrain(k,S3Idcs));
end

%Get Total Delta Parameter
S2 = S2fracs.*PS21backwards;
S3 = S3fracs.*PS32backwards;
TotalDelta = S2 + S3;
%Totaldeltas(ky) = TotalDelta(end);


%Get Delta Parameter Electron Holes
% PlotDeltaElectronHoles(n_flashes,P_totals,y_totaltrain,S0IdcsH0,S1IdcsH0,S2IdcsH0,S3IdcsH0,...
%                                     S0IdcsH1,S1IdcsH1,S2IdcsH1,S3IdcsH1,S0IdcsH2,S1IdcsH2,S2IdcsH2,S3IdcsH2,...
%                                     S0IdcsH3,S1IdcsH3,S2IdcsH3,S3IdcsH3,S0Idcs,S1Idcs,S2Idcs,S3Idcs);
%Fourier Transform:
% Fs = 1;
% nfft = 2^nextpow2(numel(Stotal));
% L = nfft;   
% Y = fft(detrend(Stotal),nfft);
% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;
% plot(f(2:end),P1(2:end)/max(P1(2:end)),'color','black','linewidth',4)
close all

x = zeros(4,4,n_flashes);
for i = 1:n_flashes
    x(1,1,i) = PS0miss(i);
    x(1,2,i) = PS1miss(i);
    x(1,3,i) = PS2miss(i);
    x(1,4,i) = PS3miss(i);
    x(2,1,i) = PS02doublehit(i); 
    x(2,2,i) = PS13doublehit(i); 
    x(2,3,i) = PS20doublehit(i); 
    x(2,4,i) = PS31doublehit(i); 
    x(3,1,i) = 0; 
    x(3,2,i) = 0; 
    x(3,3,i) = S2(i); 
    x(3,4,i) = S3(i); 
    x(4,1,i) = 0; 
    x(4,2,i) = 0; 
    x(4,3,i) = 0; 
    x(4,4,i) = 0; 
end 
x(isnan(x)) = 0;
S_initial(1) = S0fracs(1);
S_initial(2) = S1fracs(1);
S_initial(3) = S2fracs(1);
S_initial(4) = S3fracs(1);
S_initial(5) = 0;
S_initial = S_initial';
analysisType = '2';

n = 1:50;
fittingfxn = @(e,n) e(1)*PS0miss(n) + e(2)*PS1miss(n) + e(3)*PS2miss(n) + e(4)*PS3miss(n); 
fxn = @(e) sum((fittingfxn(e,n) - Totalmiss).^2);      
e = fmincon(fxn,[0.0034,0.3920,1,1]);           

n = 1:50;
fittingfxn = @(e,n) e(1)*(S0fracs.*PS0miss(n)) + e(2)*(S1fracs.*PS1miss(n)) + e(3)*(S2fracs.*PS2miss(n)) + e(4)*(S3fracs.*PS3miss(n)); 
fxn = @(e) sum((fittingfxn(e,n) - Totalmiss).^2);      
e = fmincon(fxn,[0.0034,0.3920,1,1]);           
    

O2_sim = VZAD_forward_varrying_parameters_with_S_states(x, S_initial, analysisType);
O2_sim = O2_sim/mean(O2_sim);
figure; 
plot(1:length(O2_sim), O2_sim, '-o','color','black')

% end
% figure; plot(1:length(O2_sim), O2_sim, '-o')
plot_pq_redox_state(species, ys, ts)
plot_Qb_site(species, ys, ts)
plot_Qa(species, ys, ts)
plot_Qbrrs(species, ys, ts)
plot_H_species(species, ys, ts)
O2_length = length(O2_sim);
O2_avg = sum(O2_sim)/(O2_length);  
[O2_sim] = (O2_sim/O2_avg);

end





