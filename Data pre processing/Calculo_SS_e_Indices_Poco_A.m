% close all
clc

%----------------------------------------------|----------------------------------------------
%%   -------------------------   <<<   Definindo parametros   >>>   ---------------------   %%
%----------------------------------------------|----------------------------------------------
%   ----------------   >>>   Definindo o que plotar
plot_ss = 1; 
plot_pca = 0; INTERVALOX = [118 122];

plot_rop = 0;
plot_Map = 1;
plot_MapSS = 1;
adimensionaliza = 0;
adimensionaliza_SWOB = 0;
adimensionaliza_RPM = 0;


plot_MapPCA = 1;                                                 %fig2
PCA_normaliza = 1;                                               %fig2
PCA_normaliza_padrao = 1;                                        %fig2
PCA_normaliza_log = 1;
PCA_por_partes = 0;                                              %fig2

PCA_auto_valor = 0;                                              %fig3
PCA_acomp_auto_vetor = 0;                                        %fig4
size_PCA = 1*720;    % horas
intervalo_vazio = 0; 
	
%   ----------------   >>>   Criando cuts:

seguranca_ini = 50;      % Numero de passo_tempos (BitOnBot = Perf) a pular antes do inicio do cut
seguranca_fim = 50;      % Numero de passo_tempos (BitOnBot = Perf) a pular antes do final do cut
tamanho_vetor_PCA = 50;

%   ----------------   >>>   Calculando SSs

tempo_inicio_perf = 49;
tempo_fim_perf = 136;







%% -------------------------------------------------------------------------------------------------

janela_ssSUP = 50;                  % Janela para a obtenção dos valores de max, min e mean
janela_ssDOWN = 2*janela_ssSUP;       % Janela para a obtenção dos valores de max, min e mean
janela_ssDOWN2 = 30;
janela_ropSUP = 50;

%% -------------------------------------------------------------------------------------------------

MM = 0;

if MM == 0
    Map_RPMmm = 0;
    Map_STORmm = 0;
    Map_SWOBmm = 0;
else
    Map_RPMmm = 1;
    Map_STORmm = 1;
    Map_SWOBmm = 1;
end

janela_mmRPM = 20;                  % Janela para o cálculo dos valores da media movel janelada
janela_mmSTOR = 20;                 % Janela para o cálculo dos valores da media movel janelada
janela_mmSWOB = 20;                 % Janela para o cálculo dos valores da media movel janelada


%   ----------------  >>>   Plota mapa de SS

tempo_maxA = 121.1;
depth_maxA = 5021.1;

passo_tempo = 10;
passo_depth = 50;

Map_intervalo_tempo = 1;
Map_intervalo_depth = 0;
Map_SS_RPM_DOWN = 1;
Map_SS_STOR_SUP = 0;

%   ----------------   >>>   DEFINE VETORES:
%   ---   LAS
BitOnBot_GERAL = LAS_RM_TIME_COBTM;
SUP_ElapsedTime = LAS_RM_TIME_HOUR;
SUP_RPM = LAS_RM_TIME_RPM;
SUP_STOR = LAS_RM_TIME_STOR;
SUP_SWOB = LAS_RM_TIME_SWOB;
SUP_DEPTH = LAS_RM_TIME_DEPTH;
SUP_TFLO = LAS_RM_TIME_TFLO;
SUP_VIB_TOR = LAS_RM_TIME_VIB_TOR;

SUP_GR_CAL = LAS_RM_TIME_GR_CAL;    % Calibrated Gamma Ray
% SUP_ECD = LAS_ECD;                % Equivalent Circulating Density
SUP_SPPA = LAS_RM_TIME_SPPA;        % Standpipe Pressure

%   ---   DOWN
DOWN_ElapsedTime_sync = R6K_ElapsedTime_sync;
DOWN_RPMMax = R6K_RPMMax;
DOWN_RPMMin = R6K_RPMMin;
DOWN_RPMMean = R6K_RPMMean;


%   ----------------   >>>   BitOnBot

Perf = 0;    %Valor Bit on Bottom para indicar Perfuração

for i=1:length(BitOnBot_GERAL)
    if BitOnBot_GERAL(i) ~= Perf
        BitOnBot_GERAL(i) = NaN;
    end
end


%----------------------------------------------|----------------------------------------------
%%   ----------------------------   <<<   Definindo CUTs   >>>   ------------------------   %%
%----------------------------------------------|----------------------------------------------

clearvars count cuts_all
clc

intervalo_min = seguranca_ini+seguranca_fim;
count = 0;
cuts_all = [];

for i=1:length(BitOnBot_GERAL)
    if SUP_ElapsedTime(i) > tempo_inicio_perf

        if BitOnBot_GERAL(i) == Perf
            count = count + 1;
        end
        
        if BitOnBot_GERAL(i) ~= Perf && count >= intervalo_min
            cuts_all(size(cuts_all, 1)+1, 1) = i-1 - count + seguranca_ini;
            cuts_all(size(cuts_all, 1), 2) = i-1 - seguranca_fim;
        end
        
        if BitOnBot_GERAL(i) ~= Perf
            count = 0;
        end
    end
    
    if SUP_ElapsedTime(i) > tempo_fim_perf
        break
    end
end

for k = 1:size(cuts_all, 1)
    ini = cuts_all(k, 1);
    fim = cuts_all(k, 2);
    
    if min(DOWN_ElapsedTime_sync) > SUP_ElapsedTime(ini)
        cuts_all(k, 3) = NaN;
        cuts_all(k, 4) = NaN;
        
    elseif max(DOWN_ElapsedTime_sync) < SUP_ElapsedTime(fim)
        cuts_all(k, 3) = NaN;
        cuts_all(k, 4) = NaN;
    else
        
        j = 1;
        stop = DOWN_ElapsedTime_sync(j);
        
        while DOWN_ElapsedTime_sync(j) <= SUP_ElapsedTime(fim)
            j = j + 1;
            
            if j <= length(DOWN_ElapsedTime_sync)
                
                
                if DOWN_ElapsedTime_sync(j) <= SUP_ElapsedTime(ini)
                    cuts_all(k, 3) = j;
                end
                
                if DOWN_ElapsedTime_sync(j) <= SUP_ElapsedTime(fim)
                    cuts_all(k, 4) = j;
                end
            end
            
            if j >= length(DOWN_ElapsedTime_sync)
                j = 1;
                break
            end
        end
    end
end

inicio = 1;
for i=1:size(cuts_all, 1)
    if SUP_ElapsedTime(cuts_all(i,1)) < tempo_inicio_perf
        inicio = i+1;
        if isnan(cuts_all(i, 3))
            inicio = i+1;
        end
    end
end

final = 1;
for i=1:size(cuts_all, 1)
    
    if SUP_ElapsedTime(cuts_all(i,2)) <= tempo_fim_perf
        final = i;
    end
end

cuts_all = cuts_all(inicio:final,:);
% [SUP_ElapsedTime(cuts_all(:,[1 2])) DOWN_ElapsedTime_sync(cuts_all(:,[3 4]))]


%----------------------------------------------|----------------------------------------------
%%   -----------------------   <<<   Calculando SSs e ROP   >>>   -----------------------   %%
%----------------------------------------------|----------------------------------------------

clearvars ssRPM_DOWN ssRPM_SUP ssSTOR_SUP RPMMeanSUP_vector isOKssSUP isOKssDOWN PCA_X PCA_X_n
clearvars RPMMean_SUP

j = 1;
count = 1;

PCA_coeff1 = []; PCA_coeff2 = []; PCA_coeff3 = [];
PCA_latent = []; PCA_explained = [];

for k = 2:size(cuts_all, 1)
%     (k/size(cuts_all, 1))*100    % Plota o progresso
    
    iniSUP  = cuts_all(k, 1);
    fimSUP  = cuts_all(k, 2);
    iniDOWN = cuts_all(k, 3);
    fimDOWN = cuts_all(k, 4);
    
    cuts_all(k, 5) = iniSUP+janela_ssSUP/2;
    cuts_all(k, 6) = fimSUP-janela_ssSUP/2;
    
    for i=iniSUP+janela_ssSUP/2:fimSUP-janela_ssSUP/2
        
        j = iniDOWN;
        
        while DOWN_ElapsedTime_sync(j) <= SUP_ElapsedTime(i)
            j = j + 1;
            
            if j >= length(DOWN_ElapsedTime_sync)-janela_ssDOWN/2
                j = length(DOWN_ElapsedTime_sync)-janela_ssDOWN/2;
                break
            end
        end
        
        imDOWN  = j - janela_ssDOWN/2;
        iMDOWN  = j + janela_ssDOWN/2;
        
        imDOWN  = j - janela_ssDOWN2/2;
        iMDOWN  = j + janela_ssDOWN2/2;
        
        imSUP = i - janela_ssSUP/2;
        iMSUP = i + janela_ssSUP/2;
        imropSUP = i - janela_ropSUP;
        iMropSUP = i + janela_ropSUP;
        
        im_mmRPMSUP = i - janela_mmRPM/2;
        iM_mmRPMSUP = i + janela_mmRPM/2;
        im_mmSTORSUP = i - janela_mmSTOR/2;
        iM_mmSTORSUP = i + janela_mmSTOR/2;
        im_mmSWOBSUP = i - janela_mmSWOB/2;
        iM_mmSWOBSUP = i + janela_mmSWOB/2;
        
        im_mmPCASUP = i - janela_mmRPM/2;
        iM_mmPCASUP = i + janela_mmRPM/2;
        
        if im_mmRPMSUP < ini
            im_mmRPMSUP = ini;
        end
        if iM_mmRPMSUP > fimSUP
            iM_mmRPMSUP = fimSUP;
        end
        
        if im_mmSTORSUP < iniSUP
            im_mmSTORSUP = iniSUP;
        end
        if iM_mmSTORSUP > fimSUP
            iM_mmSTORSUP = fimSUP;
        end
        
        if im_mmSWOBSUP < iniSUP
            im_mmSWOBSUP = iniSUP;
        end
        if iM_mmSWOBSUP > fimSUP
            iM_mmSWOBSUP = fimSUP;
        end
        
        if im_mmPCASUP < iniSUP
            im_mmPCASUP = iniSUP;
        end
        if iM_mmPCASUP > fimSUP
            iM_mmPCASUP = fimSUP;
        end
        
        if imropSUP < iniSUP
            imropSUP = iniSUP;
        end
        if iMropSUP > fimSUP
            iMropSUP = fimSUP;
        end
        
        if imDOWN < iniDOWN
            imDOWN = iniDOWN;
        end
        if iMDOWN > fimDOWN
            iMDOWN = fimDOWN;
        end
        
        
    
        RPMMaxDOWN = max(DOWN_RPMMax(imDOWN:iMDOWN));
        RPMMinDOWN = min(DOWN_RPMMin(imDOWN:iMDOWN));
        
        RPMMaxSUP = max(SUP_RPM(imSUP:iMSUP));
        RPMMinSUP = min(SUP_RPM(imSUP:iMSUP));
        STORMaxSUP = max(SUP_STOR(imSUP:iMSUP));
        STORMinSUP = min(SUP_STOR(imSUP:iMSUP));
        
        DEPTHMax = max(SUP_DEPTH(imropSUP:iMropSUP));
        DEPTHmin = min(SUP_DEPTH(imropSUP:iMropSUP));
        
        SUP_ElapsedTimeMax = max(SUP_ElapsedTime(imropSUP:iMropSUP));
        SUP_ElapsedTimeMin = min(SUP_ElapsedTime(imropSUP:iMropSUP));
        
        
        
        if RPMMinDOWN < 0
            RPMMinDOWN = 0;
        end
        if RPMMinSUP < 0
            RPMMinSUP = 0;
        end
        if STORMinSUP < 0
            STORMinSUP = 0;
        end
        
%         RPMMeanDOWN  = mean(DOWN_RPMMean(imDOWN:iMDOWN),'omitnan');
%         RPMMeanSUP  = mean(SUP_RPM(iniSUP:fimSUP),'omitnan');
        RPMMeanSUP  = mean(SUP_RPM(imSUP:iMSUP),'omitnan');
        STORMeanSUP  = mean(SUP_STOR(iniSUP:fimSUP),'omitnan');
        
        RPMmm_SUP(i)  = mean(SUP_RPM(im_mmRPMSUP:iM_mmRPMSUP),'omitnan');
        STORmm_SUP(i)  = mean(SUP_STOR(im_mmSTORSUP:iM_mmSTORSUP),'omitnan');
        SWOBmm_SUP(i)  = mean(SUP_SWOB(im_mmSWOBSUP:iM_mmSWOBSUP),'omitnan');
        
        deltaRPMDOWN = RPMMaxDOWN - RPMMinDOWN;
        
        

        deltaRPMSUP = RPMMaxSUP - RPMMinSUP;
        deltaSTORSUP = STORMaxSUP - STORMinSUP;
        RPMMaxDiffDOWN = max(DOWN_RPMMax(imDOWN:iMDOWN)-DOWN_RPMMin(imDOWN:iMDOWN));

        ssRPM_DOWN(i, 1)  = deltaRPMDOWN/(2*RPMMeanSUP);
%         ssRPM_DOWN(i, 1)  = RPMMaxDiffDOWN/(2*RPMMeanSUP);
        ssRPM_SUP(i) = deltaRPMSUP/(2*RPMMeanSUP);
        RPMMean_SUP(i) = RPMMeanSUP;
        ssSTOR_SUP(i) = deltaSTORSUP/(2*STORMeanSUP);
        
        ropSUP(i, 1) = (DEPTHMax - DEPTHmin)/(SUP_ElapsedTimeMax - SUP_ElapsedTimeMin);
                
    end
end

clearvars Vetor_padrao
Vetor_padrao = ones(length(ssRPM_DOWN), 1);

for i=1:length(Vetor_padrao)
    if ssRPM_DOWN(i) == 0
        Vetor_padrao(i) = NaN;
    end
end
isOK_Vetor_padrao = isfinite(Vetor_padrao);

%----------------------------------------------|----------------------------------------------
%%   ------------------------   <<<   Calculando PCA no t   >>>   -----------------------   %%
%----------------------------------------------|----------------------------------------------

if plot_pca == 1
    
    %   ----------------   >>>   Define Vetor para o PCA:
    PCA_X = [SUP_RPM,...
        SUP_DEPTH,...
        SUP_STOR,...
        SUP_SWOB];
    
    PCA_ElapsedTime = SUP_ElapsedTime(1:size(PCA_X, 1));
    PCA_ElapsedTime = PCA_ElapsedTime(isOK_Vetor_padrao);
    
    for i=1:size(PCA_X, 2)
        clearvars aa
        aa = PCA_X(1:length(Vetor_padrao), i);
        PCA_X_puro(:, i) = aa(isOK_Vetor_padrao);
    end
    clearvars aa
    
    

    PCA_X_puro_PorPartes_n = [];
    PCA_coeff1 = []; PCA_coeff2 = []; PCA_coeff3 = [];
    PCA_latent = []; PCA_explained = [];
    
    for i=1+tamanho_vetor_PCA/2:size(PCA_X_puro, 1)-tamanho_vetor_PCA/2
        i/(size(PCA_X_puro, 1)-tamanho_vetor_PCA/2)
        
        PCA_X_puro_PorPartes = PCA_X_puro(i-tamanho_vetor_PCA/2:i+tamanho_vetor_PCA/2, :);
        
 %   ----------------   >>>   Normaliza Vetor para o PCA PorPartes:       
        
        for j=1:size(PCA_X_puro_PorPartes, 2)
            clearvars RAW normalizado
            RAW = PCA_X_puro_PorPartes(:,j);
            
%             if PCA_normaliza_padrao == 1
%                 normalizado = (RAW - mean(RAW, 'omitnan'))/X_PocoA_SS_m1_std(j);
%                 
%             elseif PCA_normaliza_log == 1
%                 if min(RAW) <= 0
%                     for k=1:length(RAW)
%                         if RAW(k) <= 0
%                             RAW(k) = NaN;
%                         end
%                     end
%                 end
%                 normalizado = log(RAW) - log(std(RAW, 'omitnan'));
%             else
%                 normalizado = (RAW - mean(RAW, 'omitnan'))/std(RAW, 'omitnan');
%             end
            
            normalizado = (RAW - mean(RAW, 'omitnan'))/std(RAW, 'omitnan');
            
            PCA_X_puro_PorPartes_n(:,j) = normalizado;
            clearvars RAW normalizado
        end
        
        
        
        [coeff,score,latent,tsquared,explained,mu] = pca(PCA_X_puro_PorPartes_n);
        
        PCA_coeff1(:, i) = coeff(:, 1);
        PCA_coeff2(:, i) = coeff(:, 2);
        PCA_coeff3(:, i) = coeff(:, 3);
        PCA_latent(:, i) = latent;
        PCA_explained(:, i) = explained;
        
        clearvars coeff score latent tsquared explained mu
    
    end
end






%----------------------------------------------|----------------------------------------------
%%   ----------------------   <<<   Plotando SS, ROP e PCA   >>>   ----------------------   %%
%----------------------------------------------|----------------------------------------------
% INTERVALOX = [115 136];
if plot_ss == 1
    
    figure('Color', 'w')
    
    subplot(5,1,1)
    plot(SUP_ElapsedTime(isOK_Vetor_padrao), SUP_RPM(isOK_Vetor_padrao),'.',...
        SUP_ElapsedTime(isOK_Vetor_padrao), SUP_STOR(isOK_Vetor_padrao),'.',...
        SUP_ElapsedTime(isOK_Vetor_padrao), RPMMean_SUP(isOK_Vetor_padrao),'.')
    legend('LAS - RPM', 'LAS - STOR', 'LAS - RPMMeam')
    grid on
    ylim([0 200])
    xlim(INTERVALOX)
    
    subplot(5,1,2)
    plot(DOWN_ElapsedTime_sync, DOWN_RPMMax,'.',...
        DOWN_ElapsedTime_sync, DOWN_RPMMin,'.')
    legend('BBHD - RPMMax', 'BBHD - RPMMin')
    grid on
    xlim(INTERVALOX); ylim([0 600]);
    
    subplot(5,1,3)
    plot(SUP_ElapsedTime(1:length(ssRPM_DOWN)), ssRPM_DOWN,...
         SUP_ElapsedTime(1:length(BitOnBot_GERAL)), BitOnBot_GERAL, 'k.')
    legend('SS', 'LAS - BitOnBot')
    grid on
    xlim(INTERVALOX)
    
    subplot(5,1,4)
    plot(SUP_ElapsedTime(1:length(ssSTOR_SUP)), ssSTOR_SUP,...
        SUP_ElapsedTime(1:length(BitOnBot_GERAL)), BitOnBot_GERAL, 'k.')
    legend('SS STOR', 'LAS - BitOnBot')
    grid on
    xlim(INTERVALOX)
    
    subplot(5,1,5)
    plot(SUP_ElapsedTime(1:length(ropSUP)), ropSUP,...
        SUP_ElapsedTime(1:length(BitOnBot_GERAL)), BitOnBot_GERAL, 'k.')
    legend('ROP', 'LAS - BitOnBot')
    grid on
    xlim(INTERVALOX)
    
end

if plot_rop == 1
    figure('Color', 'w')
    plot(SUP_ElapsedTime(1:length(ropSUP)), ropSUP)
    grid on
    ylabel('m/h')
    xlim(INTERVALOX)
end


if plot_pca == 1
    figure('Color', 'w')
   
    subplot(5,1,1)
    plot(DOWN_ElapsedTime_sync, DOWN_RPMMax,'.',...
        DOWN_ElapsedTime_sync, DOWN_RPMMin,'.')
    legend('RPM - DOWN')
    ylim([0 500]); grid on
    grid on
    xlim(INTERVALOX)
    
    subplot(5,1,2)
    plot(SUP_ElapsedTime(1:length(ssRPM_SUP)), ssRPM_SUP,...
        SUP_ElapsedTime(1:length(ssSTOR_SUP)), ssSTOR_SUP,...
        SUP_ElapsedTime(1:length(ssRPM_DOWN)), ssRPM_DOWN,...
        SUP_ElapsedTime(1:length(BitOnBot_GERAL)), BitOnBot_GERAL, 'k.')
    legend('ssRPM SUP', 'ssSTOR SUP', 'ssRPM DOWN', 'BitOnBot SUP')
    ylim([0 3]); grid on
    xlim(INTERVALOX)
    
    subplot(5,1,3)
    plot(PCA_ElapsedTime(1:length(PCA_ElapsedTime)-tamanho_vetor_PCA/2), PCA_explained(1, :), 'k.')
    ylabel('AV 1 norm')
    xlim(INTERVALOX); grid on
    
    subplot(5,1,4)
    plot(PCA_ElapsedTime(1:length(PCA_ElapsedTime)-tamanho_vetor_PCA/2), PCA_explained(2, :), 'b.')
    ylabel('AV 2 norm')
    xlim(INTERVALOX); grid on
    
    subplot(5,1,5)
    plot(PCA_ElapsedTime(1:length(PCA_ElapsedTime)-tamanho_vetor_PCA/2), PCA_explained(3, :), 'b.')
    ylabel('AV 3 norm')
    xlim(INTERVALOX); grid on
end




%----------------------------------------------|----------------------------------------------
%%   -----------------------   <<<   Redimenciona Vetores   >>>   -----------------------   %%
%----------------------------------------------|----------------------------------------------
clearvars ssRPM_DOWN2 ssRPM_SUP2 isOKssDOWN2 isOKssSUP2

ssRPM_DOWN2 = ssRPM_DOWN;
for i=1:length(ssRPM_DOWN2)
    if ssRPM_DOWN2(i) == 0
        ssRPM_DOWN2(i) = NaN;
    end
end
isOKssDOWN2 = isfinite(ssRPM_DOWN2);

ssRPM_SUP2 = ssRPM_SUP;
for i=1:length(ssRPM_SUP2)
    if ssRPM_SUP2(i) == 0
        ssRPM_SUP2(i) = NaN;
    end
end
isOKssSUP2 = isfinite(ssRPM_SUP2);





%%
clearvars RPM_SUP_DOWN HOUR_SUP_DOWNS_DOWN SWOB_SUP_DOWN STOR_SUP_DOWN RPMmm_SUP_DOWN STORmm_SUP_DOWN SWOBmm_SUP_DOWN
clearvars SS_RPM_DOWN_SUP SS_STOR_SUP

RPM_SUP_DOWN = SUP_RPM(isOKssDOWN2);
HOUR_SUP_DOWN = SUP_ElapsedTime(isOKssDOWN2);
DEPTH_SUP_DOWN = SUP_DEPTH(isOKssDOWN2);
SWOB_SUP_DOWN = SUP_SWOB(isOKssDOWN2);
STOR_SUP_DOWN = SUP_STOR(isOKssDOWN2);
TFLO_SUP_DOWN = SUP_TFLO(isOKssDOWN2);
VIB_TOR_SUP_DOWN = SUP_VIB_TOR(isOKssDOWN2);
ROP_SUP = ropSUP(isOKssDOWN2);

GR_CAL_SUP_DOWN = SUP_GR_CAL(isOKssDOWN2);    % Calibrated Gamma Ray
SPPA_SUP_DOWN = SUP_SPPA(isOKssDOWN2);        % Standpipe Pressure


RPMmm_SUP_DOWN = RPMmm_SUP(isOKssDOWN2);
STORmm_SUP_DOWN = STORmm_SUP(isOKssDOWN2);
SWOBmm_SUP_DOWN = SWOBmm_SUP(isOKssDOWN2);

SS_RPM_DOWN_SUP = ssRPM_DOWN2(isOKssDOWN2);
SS_STOR_SUP = ssSTOR_SUP(isOKssDOWN2)*4.5;

i=1;
while HOUR_SUP_DOWN(i) <= tempo_inicio_perf
    i = i+1;
end

j=1;
while HOUR_SUP_DOWN(j) <= tempo_fim_perf
    j = j+1;
    if j > length(HOUR_SUP_DOWN)
        j = length(HOUR_SUP_DOWN);
        break
    end
end

RPM_SUP_DOWN = RPM_SUP_DOWN(i:j);
HOUR_SUP_DOWN = HOUR_SUP_DOWN(i:j);
DEPTH_SUP_DOWN = DEPTH_SUP_DOWN(i:j);
SWOB_SUP_DOWN = SWOB_SUP_DOWN(i:j);
STOR_SUP_DOWN = STOR_SUP_DOWN(i:j);
TFLO_SUP_DOWN = TFLO_SUP_DOWN(i:j);
VIB_TOR_SUP_DOWN = VIB_TOR_SUP_DOWN(i:j);
ROP_SUP_DOWN = ROP_SUP(i:j);

GR_CAL_SUP_DOWN = GR_CAL_SUP_DOWN(i:j);    % Calibrated Gamma Ray
% ECD_SUP_DOWN = ECD_SUP_DOWN(i:j);       % Equivalent Circulating Density
SPPA_SUP_DOWN = SPPA_SUP_DOWN(i:j);      % Standpipe Pressure

RPMmm_SUP_DOWN = RPMmm_SUP_DOWN(i:j);
STORmm_SUP_DOWN = STORmm_SUP_DOWN(i:j);
SWOBmm_SUP_DOWN = SWOBmm_SUP_DOWN(i:j);

SS_RPM_DOWN_SUP = SS_RPM_DOWN_SUP(i:j);
SS_STOR_SUP = SS_STOR_SUP(i:j);


DIMENSOES   =  [HOUR_SUP_DOWN,...
                RPM_SUP_DOWN,...
                SWOB_SUP_DOWN,...
                STOR_SUP_DOWN,...
                ROP_SUP_DOWN,...
                SS_STOR_SUP.',...
                SPPA_SUP_DOWN];


dim_size = size(DIMENSOES,2);
a = [];
a = [DIMENSOES SS_RPM_DOWN_SUP HOUR_SUP_DOWN];
SS_RPM_DOWN_SUP = [];
HOUR_SUP_DOWN = [];
DIMENSOES = [];

count = 1;

for i=1:size(a, 1)
    if (isfinite(sum(a(i,:)))) && min(a(i, :)) > 0
        DIMENSOES(count, :) = a(i, 1:dim_size);
        SS_RPM_DOWN_SUP(count,1) = a(i, dim_size + 1);
        HOUR_SUP_DOWN(count,1) = a(i, dim_size + 2);
        count = count+1;
    end
end





%----------------------------------------------|----------------------------------------------
%%   -------------------------   <<<   Cria Intervalos   >>>   --------------------------   %%
%----------------------------------------------|----------------------------------------------

clearvars INTERVALO_TIME INTERVALO_DEPTH legenda depth_atual


INTERVALO_TIME = [tempo_inicio_perf  tempo_fim_perf;...
    tempo_inicio_perf  tempo_maxA;...
    tempo_maxA         tempo_fim_perf];

depth_min = min(DEPTH_SUP_DOWN);
depth_atual = depth_min;
INTERVALO_DEPTH = [min(DEPTH_SUP_DOWN) max(DEPTH_SUP_DOWN);...
    min(DEPTH_SUP_DOWN) depth_maxA;...
    depth_maxA          max(DEPTH_SUP_DOWN)];

if intervalo_vazio == 1
    INTERVALO_DEPTH = [];
    INTERVALO_TIME = [];
end

tempo_atual = tempo_inicio_perf;
if Map_intervalo_tempo == 1
    count = size(INTERVALO_TIME, 1) + 1;
    
    while tempo_atual < tempo_fim_perf
        INTERVALO_TIME(count, 1) = tempo_atual;
        tempo_atual = tempo_atual + passo_tempo;
        
        if tempo_atual > tempo_fim_perf
            tempo_atual = tempo_fim_perf;
        end
        
        INTERVALO_TIME(count, 2) = tempo_atual;
        count = count + 1;
    end
end


if Map_intervalo_depth == 1
    count = size(INTERVALO_DEPTH, 1) + 1;
    while depth_atual < depth_maxA
        INTERVALO_DEPTH(count, 1) = depth_atual;
        depth_atual = depth_atual + passo_depth;
        
        if depth_atual > depth_maxA
            depth_atual = depth_maxA;
        end
        
        INTERVALO_DEPTH(count, 2) = depth_atual;
        count = count + 1;
    end
    
    while depth_atual < 5027
        INTERVALO_DEPTH(count, 1) = depth_atual;
        depth_atual = depth_atual + passo_depth;
        
        if depth_atual > 5027
            depth_atual = 5027;
        end
        
        INTERVALO_DEPTH(count, 2) = depth_atual;
        count = count + 1;
    end
end


%----------------------------------------------|----------------------------------------------
%%   ----------------------------   <<<   Plota Mapas   >>>   ---------------------------   %%
%----------------------------------------------|----------------------------------------------

if Map_intervalo_tempo == 1
    INTERVALO = INTERVALO_TIME;
end
if Map_intervalo_depth == 1
    INTERVALO = INTERVALO_DEPTH;
end

clearvars DIMENCOES_n

if plot_MapPCA == 1

    for i=1:size(DIMENSOES, 2)
        clearvars RAW normalizado coeff_n score_n latent_n tsquared_n explained_n mu_n
        RAW = DIMENSOES(:,i);
        
        if PCA_normaliza_padrao == 1
            normalizado = (RAW - mean(RAW, 'omitnan'))/X_PocoA_SS_m1_std(i);
            
        elseif PCA_normaliza_log == 1
            if min(RAW) <= 0
                for k=1:length(RAW)
                    if RAW(k) <= 0
                        RAW(k) = NaN;
                    end
                end
            end
            
            normalizado = log(RAW) - log(std(RAW, 'omitnan'));
        else
            normalizado = (RAW - mean(RAW, 'omitnan'))/std(RAW, 'omitnan');
        end

        DIMENSOES_n(:,i) = normalizado;
    end
    
    
    [coeff_n,score_n,latent_n,tsquared_n,explained_n,mu_n] = pca(DIMENSOES_n); % , 'Rows','pairwise');
    
    x = 1:size(INTERVALO, 1);
    y = 1:size(DIMENSOES, 2);
    [X,Y] = meshgrid(x,y);
    
    Mesh_AutoVetores1 = zeros(size(DIMENSOES, 2), size(INTERVALO, 1));
    Mesh_AutoVetores2 = zeros(size(DIMENSOES, 2), size(INTERVALO, 1));
    Mesh_AutoVetores3 = zeros(size(DIMENSOES, 2), size(INTERVALO, 1));
    
    Mesh_Score = [];
    
    Mesh_AutoValores = zeros(size(DIMENSOES, 2), size(INTERVALO, 1));
    Mesh_AutoPorcentagem = zeros(size(DIMENSOES, 2), size(INTERVALO, 1));
    
end

%%
if plot_Map == 1
    clearvars legenda
    
%   ----------------   >>>   Cria Figuras    
    if plot_MapSS == 1
        if adimensionaliza == 1
            if Map_RPMmm == 0
                fig1 = figure('Name', 'MapaSS 3D Adimensionalizado', 'NumberTitle','on', 'units','normalized','outerposition',[0 0.04 1 0.91]);
                
            elseif Map_RPMmm == 1
                figure('Name', 'MapaSS 3D Adimensionalizado mm', 'NumberTitle','on', 'units','normalized','outerposition',[0 0.04 1 0.91]);
            end
        else
            if Map_RPMmm == 0
                fig1 = figure('Name', 'MapaSS 3D Padrão', 'NumberTitle','on', 'units','normalized','outerposition',[0 0.04 1 0.91]);
            elseif Map_RPMmm == 1
                fig1 = figure('Name', 'MapaSS 3D Padrão mm', 'NumberTitle','on', 'units','normalized','outerposition',[0 0.04 1 0.91]);
            end
        end
    end
    
    if plot_MapPCA == 1
        
        fig2 = figure('Name', 'MapaSS 3D PCA', 'NumberTitle','on', 'units','normalized','outerposition',[0 0.04 1 0.91]);
        if PCA_por_partes == 1
            if PCA_auto_valor == 1
                fig3 = figure('Name', 'PCA AutoValor', 'NumberTitle','on', 'units','normalized','outerposition',[0 0.04 1 0.91]);
            end
            
        end
        if PCA_acomp_auto_vetor == 1
            fig4 = figure('Name', 'PCA AutoVetor', 'NumberTitle','on', 'units','normalized','outerposition',[0 0.04 1 0.91]);
        end
    end
    
    count = 1;
    count_ant = 1;
    
    for i=1:size(INTERVALO, 1)
        clearvars RPM_SUP_DOWN2 HOUR_SUP_DOWNS_DOWN2 SWOB_SUP_DOWN2 STOR_SUP_DOWN2
        clearvars RPMmm_SUP_DOWN2 STORmm_SUP_DOWN2 SWOBmm_SUP_DOWN2
        clearvars SS_RPM_DOWN_SUP2 SS_STOR_SUP2
        clearvars SWOB RPM STOR SS DEPTH
        
        ini = INTERVALO(i, 1);
        fim = INTERVALO(i, 2);
        
        nome_ini = char(string(ini));
        nome_fim = char(string(fim));
        
        pos_ini = 1;
        pos_fim = 1;
        
        if Map_intervalo_tempo == 1
            while HOUR_SUP_DOWN(pos_ini) <= ini
                pos_ini = pos_ini + 1;
                if pos_ini > length(HOUR_SUP_DOWN)
                    pos_ini = length(HOUR_SUP_DOWN);
                    break
                end
            end
            
            while HOUR_SUP_DOWN(pos_fim) <= fim
                pos_fim = pos_fim + 1;
                if pos_fim >= length(HOUR_SUP_DOWN)
                    pos_fim = length(HOUR_SUP_DOWN);
                    break
                end
            end
        end
        
        if Map_intervalo_depth == 1
            while DEPTH_SUP_DOWN(pos_ini) <= ini
                pos_ini = pos_ini + 1;
                if pos_ini > length(DEPTH_SUP_DOWN)
                    pos_ini = length(DEPTH_SUP_DOWN);
                    break
                end
            end
            
            while DEPTH_SUP_DOWN(pos_fim) <= fim
                pos_fim = pos_fim + 1;
                if pos_fim >= length(DEPTH_SUP_DOWN)
                    pos_fim = length(DEPTH_SUP_DOWN);
                    break
                end
            end
        end
        
        
        if PCA_por_partes == 1
            clearvars DIMENCOES_PorPart DIMENCOES_PorPart_n

            DIMENCOES_PorPart = DIMENSOES(pos_ini:pos_fim,:);
            
            for k=1:size(DIMENCOES_PorPart, 2)
                clearvars RAW normalizado
                RAW = DIMENCOES_PorPart(:,k);
                
                if PCA_normaliza_padrao == 1
                    normalizado = (RAW - mean(RAW, 'omitnan'))/X_PocoA_SS_m1_std(i);
                    
                elseif PCA_normaliza_log == 1
                    if min(RAW) <= 0
                        for o=1:length(RAW)
                            if RAW(o) <= 0
                                RAW(o) = NaN;
                            end
                        end
                    end
                    normalizado = log(RAW) - log(std(RAW, 'omitnan'));
                else
                    normalizado = (RAW - mean(RAW, 'omitnan'))/std(RAW, 'omitnan');
                end
                
                DIMENCOES_PorPart_n(:, k) = normalizado;
            end
            
            [coeff_PorPart_n,score_PorPart_n,latent_PorPart_n,tsquared_PorPart_n,explained_PorPart_n,mu_PorPart_n] = pca(DIMENCOES_PorPart_n); 
        end
        
        
        RPM_SUP_DOWN2 = RPM_SUP_DOWN(pos_ini:pos_fim);
        HOUR_SUP_DOWN2 = HOUR_SUP_DOWN(pos_ini:pos_fim);
        DEPTH_SUP_DOWN2 = DEPTH_SUP_DOWN(pos_ini:pos_fim);
        SWOB_SUP_DOWN2 = SWOB_SUP_DOWN(pos_ini:pos_fim);
        STOR_SUP_DOWN2 = STOR_SUP_DOWN(pos_ini:pos_fim);
        RPMmm_SUP_DOWN2 = RPMmm_SUP_DOWN(pos_ini:pos_fim);
        STORmm_SUP_DOWN2 = STORmm_SUP_DOWN(pos_ini:pos_fim);
        SWOBmm_SUP_DOWN2 = SWOBmm_SUP_DOWN(pos_ini:pos_fim);
        
        SS_RPM_DOWN_SUP2 = SS_RPM_DOWN_SUP(pos_ini:pos_fim);
        SS_STOR_SUP2 = SS_STOR_SUP(pos_ini:pos_fim);
        
        DEPTH = DEPTH_SUP_DOWN2;
        
        if Map_RPMmm == 1
            if Map_intervalo_tempo == 1
                RPM = RPMmm_SUP_DOWN2;
            elseif Map_intervalo_depth == 1
                RPM = mean(RPMmm_SUP_DOWN2);
            end
        else
            if Map_intervalo_tempo == 1
                RPM = RPM_SUP_DOWN2;
            elseif Map_intervalo_depth == 1
                RPM = mean(RPM_SUP_DOWN2);
            end
        end
        
        if Map_STORmm == 1
            if Map_intervalo_tempo == 1
                STOR = STORmm_SUP_DOWN2;
            elseif Map_intervalo_depth == 1
                STOR = mean(STORmm_SUP_DOWN2);
            end
        else
            if Map_intervalo_tempo == 1
                STOR = STOR_SUP_DOWN2;
            elseif Map_intervalo_depth == 1
                STOR = mean(STOR_SUP_DOWN2);
            end
        end
        
        if Map_SWOBmm == 1
            if Map_intervalo_tempo == 1
                SWOB = SWOBmm_SUP_DOWN2;
            elseif Map_intervalo_depth == 1
                SWOB = mean(SWOBmm_SUP_DOWN2);
            end
        else
            if Map_intervalo_tempo == 1
                SWOB = SWOB_SUP_DOWN2;
            elseif Map_intervalo_depth == 1
                SWOB = mean(SWOB_SUP_DOWN2);
            end
        end
        
        if Map_SS_RPM_DOWN == 1
            if Map_intervalo_tempo == 1
                SS = SS_RPM_DOWN_SUP2;
            elseif Map_intervalo_depth == 1
                SS = max(SS_RPM_DOWN_SUP2);
            end
        end
        
        if Map_SS_STOR_SUP == 1
            if Map_intervalo_tempo == 1
                SS = SS_STOR_SUP2;
            elseif Map_intervalo_depth == 1
                SS = mean(SS_STOR_SUP2);
            end
        end
        
        
        if adimensionaliza == 1
            count = 1;
            for k=1:length(SWOB)
                while Lenght_m(count) < DEPTH(k)
                    count = count + 1;
                    if count > length(Lenght_m)
                        count = length(Lenght_m);
                    end
                end
                
                count_ant = count;
                
                if adimensionaliza_SWOB == 1
                    SWOB(k) = SWOB(k)*(17.5*0.5*0.0254)/Stiffness_Nm(count);
                end
                if adimensionaliza_RPM == 1
                    RPM(k) = RPM(k)/Nfrequency_Hz(count);
                end
            end
        end
        
        %   ----------------   >>>   Define Legendas
        leg = [nome_ini  ' - ' nome_fim];
        for j=1:length(leg)
            legenda(i,j) = leg(j);
        end
        
        
        if plot_MapSS == 1
            figure(fig1)
            hold on
            scatter3(RPM, SWOB, STOR, [20], SS, 'filled')
            colormap(jet);
            caxis([0.5 2.7])
            
        end
        
        if plot_MapPCA == 1
            
            if PCA_por_partes == 1
                
                PC1_PorPart_n = score_PorPart_n(:,1); PC11_PorPart_n = PC1_PorPart_n;
                PC2_PorPart_n = score_PorPart_n(:,2); PC22_PorPart_n = PC2_PorPart_n;
                PC3_PorPart_n = score_PorPart_n(:,3); PC33_PorPart_n = PC3_PorPart_n;
                
                PCA_Autovalores = ones(size(PC11_PorPart_n, 1),length(latent_PorPart_n));
                
                for k=1:length(latent_PorPart_n)
                    PCA_Autovalores(:, k) = latent_PorPart_n(k)*PCA_Autovalores(:, k).';
                end
                
                
                
            else
                
                PC1_n = score_n(:,1); PC11_n = PC1_n(pos_ini:pos_fim);
                PC2_n = score_n(:,2); PC22_n = PC2_n(pos_ini:pos_fim);
                PC3_n = score_n(:,3); PC33_n = PC3_n(pos_ini:pos_fim);
                
            end
            
            if PCA_por_partes == 1
                figure(fig2);
                hold on
                scatter3(PC11_PorPart_n, PC22_PorPart_n, PC33_PorPart_n, [20], SS, 'filled')
                colormap(jet);
                colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
                
                if PCA_auto_valor == 1
                    figure(fig3);
                    hold on
                    for k=1:size(PCA_Autovalores, 2)
                        subplot(size(PCA_Autovalores, 2), 1, k)
                        hold on
                        plot(HOUR_SUP_DOWN2, PCA_Autovalores(:,k))
                    end
                end
                
            else
                figure(fig2);
                hold on
                scatter3(PC11_n, PC22_n, PC33_n, [20], SS, 'filled')
                colormap(jet);
                colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
            end
        end
        
        if plot_MapSS == 1
            figure(fig1);
            hold on
            
            colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
            legend(legenda)
            xlabel('RPM'); ylabel('SWOB'); zlabel('STOR')
            
            if adimensionaliza == 1
                if adimensionaliza_SWOB == 1
                    ylim([0 0.07]); ylabel('(WOB*r_B)/K_e_q');
                else
                    ylim([0 120])
                end
                
                if adimensionaliza_RPM == 1
                    xlim([0 1800]); xlabel('RPM/W_n');
                else
                    xlim([0 260])
                end
                zlim([0 60])
                
                if Map_RPMmm == 0
                    title('Adimensionalizado')
                else
                    title('Adimensionalizado mm')
                end
            else
                xlim([0 260]); ylim([0 120]); zlim([0 60])
                if Map_RPMmm == 0
                    title('Padrão')
                else
                    title('Padrão mm')
                end
            end
            
            hold off
        end
        
        if plot_MapPCA == 1
            figure(fig2);
            hold on
            
            colormap(jet);
            colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
            legend(legenda)
            title('Padrão PCA')
            
            if PCA_por_partes == 1
                xlabel(['PC1n: ' + string(Mesh_AutoPorcentagem(1,1)) + '%']); ylabel(['PC2n: ' + string(Mesh_AutoPorcentagem(2,1)) + '%']); zlabel(['PC4n: ' + string(Mesh_AutoPorcentagem(3,1)) + '%'])
                title('Padrão PCA PorPartes')
            elseif PCA_por_partes == 0
                xlabel(['PC1: ' + string(explained_n(1)) + '%']); ylabel(['PC2: ' + string(explained_n(2)) + '%']); zlabel(['PC4: ' + string(explained_n(3)) + '%'])
            end
            
            hold off
        end
    end

    if PCA_acomp_auto_vetor == 1
        
        Mesh_AutoVetores1_fix = [zeros(size(DIMENSOES, 2), size_PCA)];
        Mesh_AutoVetores2_fix = [zeros(size(DIMENSOES, 2), size_PCA)];
        Mesh_AutoVetores3_fix = [zeros(size(DIMENSOES, 2), size_PCA)];
        Mesh_Score_fix = [];
        Mesh_AutoValores_fix = [zeros(size(DIMENSOES, 2), size_PCA)];
        Mesh_AutoPorcentagem_fix = [zeros(size(DIMENSOES, 2), size_PCA)];
               
        %         for i=1+size_PCA/2:size_PCA:size(DIMENCOES, 1)-size_PCA/2
        
        for i=1+size_PCA:1:size(DIMENSOES, 1)-size_PCA
            
%             DIMENCOES_Porpart_fix = DIMENCOES(i-size_PCA/2: i+size_PCA/2, :);

            DIMENCOES_Porpart_fix = DIMENSOES(i-size_PCA: i, :);
            
            [coeff_fix,score_fix,latent_fix,tsquared_fix,explained_fix,mu_fix] = pca(DIMENCOES_Porpart_fix);
            
            Mesh_AutoVetores1_fix = [Mesh_AutoVetores1_fix, coeff_fix(:, 1)];
            Mesh_AutoVetores2_fix = [Mesh_AutoVetores2_fix, coeff_fix(:, 2)];
            Mesh_AutoVetores3_fix = [Mesh_AutoVetores3_fix, coeff_fix(:, 3)];
            %     Mesh_Score_fix = [Mesh_Score_fix; score_fix];
            Mesh_AutoValores_fix = [Mesh_AutoValores_fix,  latent_fix];
            Mesh_AutoPorcentagem_fix = [Mesh_AutoPorcentagem_fix, explained_fix];
            
        end
        
        x = 1:size(Mesh_AutoVetores1_fix, 2);
        y = 1:size(DIMENSOES, 2);
        [X_fix,Y_fix] = meshgrid(x,y);
        
        clearvars Mesh_AutoVetores1_fix_n Mesh_AutoVetores2_fix_n Mesh_AutoVetores3_fix_n
        for i=1:size(Mesh_AutoVetores1_fix, 2)
            [a b] = max(abs(Mesh_AutoVetores1_fix(:, i)));
            Mesh_AutoVetores1_fix_n(:, i) = Mesh_AutoVetores1_fix(:, i)/Mesh_AutoVetores1_fix(b, i);
            [a b] = max(abs(Mesh_AutoVetores2_fix(:, i)));
            Mesh_AutoVetores2_fix_n(:, i) = Mesh_AutoVetores2_fix(:, i)/max(Mesh_AutoVetores2_fix(b, i));
            [a b] = max(abs(Mesh_AutoVetores3_fix(:, i)));
            Mesh_AutoVetores3_fix_n(:, i) = Mesh_AutoVetores3_fix(:, i)/max(Mesh_AutoVetores3_fix(b, i));
        end
        
        figure(fig4)
        
        subplot(5,1,1)
        aa = SUP_ElapsedTime(1:length(ssRPM_DOWN))
%         SSy = [zeros(size_PCA; ssRPM_DOWN(isOK_Vetor_padrao))]
        plot([zeros(size_PCA,1); ssRPM_DOWN(isOK_Vetor_padrao)], '.')
        grid on; xlim([0 length(ssRPM_DOWN(isOK_Vetor_padrao))]);
        ylabel('SS')
        title('PCA no tempo')
        
        subplot(5,1,2)
        mesh(X_fix(1:3, :), flipud(Y_fix(1:3, :)), Mesh_AutoPorcentagem_fix(1:3, :),'FaceColor','texturemap',  'EdgeColor','none')
        ylabel('%')
        colormap(jet); colorbar(); view(0, 90); xlim([1 size(X_fix, 2)]); ylim([1 3])
        
        subplot(5,1,3)
        mesh(X_fix, flipud(Y_fix), Mesh_AutoVetores1_fix_n,'FaceColor','texturemap',  'EdgeColor','none')
        ylabel('PC1')
        colormap(jet); colorbar(); view(0, 90); xlim([1 size(X_fix, 2)]); ylim([1 size(X_fix, 1)])
        
        subplot(5,1,4)
        mesh(X_fix, flipud(Y_fix), Mesh_AutoVetores2_fix_n,'FaceColor','texturemap',  'EdgeColor','none')
        ylabel('PC2')
        colormap(jet); colorbar(); view(0, 90); xlim([1 size(X_fix, 2)]); ylim([1 size(X_fix, 1)])
        
        subplot(5,1,5)
        mesh(X_fix, flipud(Y_fix), Mesh_AutoVetores3_fix_n,'FaceColor','texturemap',  'EdgeColor','none')
        ylabel('PC3')
        colormap(jet); colorbar(); view(0, 90); xlim([1 size(X_fix, 2)]); ylim([1 size(X_fix, 1)])
    end
    
end

%%

Xt_PocoA = DIMENSOES;
Xt_PocoA_SS = SS_RPM_DOWN_SUP;

Xt_PocoA_50_100_50mm = [Xt_PocoA, Xt_PocoA_SS];



