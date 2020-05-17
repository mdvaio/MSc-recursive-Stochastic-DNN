clc

SUP_X = LAS_ElapsedTime;

SUP_BitOnBotton = LAS_OBTM;

SUP_RPM = LAS_RPM;
SUP_CRPM = LAS_CRPM;

SUP_STOR = LAS_STOR;

SUP_SWOB = LAS_SWOB;

SUP_DEPTH = LAS_BIT_DEPTH;

BB_X = BHA_ElapsedTime_sync;
BB_Y1 = BHA_BB_Avg_DownholeRPM;
BB_Y2 = BHA_BB_Avg_MaxLateralAcc;

% BBHD_X = BB_X;
% BBHD_Y1 = BHA_BBHD_RPMMax;
% BBHD_Y2 = BHA_BBHD_RPMMean;
% BBHD_Y3 = BHA_BBHD_RPMMin;
% 
% BBHD_X = BB_X;
% BBHD_Y1 = BHA_BBHD_RPMMax;
% BBHD_Y2 = BHA_BBHD_RPMMean;
% BBHD_Y3 = BHA_BBHD_RPMMin;




INTERVALOX = [10 140];
% BHA_ElapsedTime_sync = BHA_ElapsedTime+23.395;



% close all
% figure
 
subplot(7,1,1)
plot(SUP_X, SUP_BitOnBotton, 'k.')
legend('LAS - BitOnBotton')
ylabel('m')
ylim([-0.2 1.2])
grid on
xlim(INTERVALOX)

subplot(7,1,2)
plot(SUP_X, SUP_RPM, 'b.',...
     SUP_X, SUP_CRPM, 'r.')
legend('LAS - RPM','LAS - Collar RPM')
ylabel('RPM')
ylim([0, 250])
grid on
xlim(INTERVALOX)

subplot(7,1,3)
plot(SUP_X, SUP_STOR, 'b.')
legend('LAS - STOR')
ylabel('1000 ft.lbf ')
ylim([0, 60])
grid on
xlim(INTERVALOX)

subplot(7,1,4)
plot(SUP_X, SUP_SWOB, 'b.')
legend('LAS - SWOB')
ylabel('1000 lbf')
ylim([0, 100])
grid on
xlim(INTERVALOX)

subplot(7,1,5)
plot(SUP_X, SUP_DEPTH, 'b.')
legend('LAS - DEPTH')
ylabel('DEPTH (m)')
grid on
xlim(INTERVALOX)

subplot(7,1,6)
plot(BB_X, BB_Y1, 'r.',...
     BB_X, BB_Y2, 'k.')
legend('BBPlug - Avg RPM', 'BBPlug - Max Lat Acell')
ylabel('g')
grid on
xlim(INTERVALOX)

% subplot(7,1,7)
% plot(BBHD_X, BBHD_Y1, '.',...
%      BBHD_X, BBHD_Y2, '.',...
%      BBHD_X, BBHD_Y3, '.')
% xlim(INTERVALOX)
% ylim([0, 500])
% grid on
% legend('BBHD - RPMMax', 'BBHD - RPMMean', 'BBHD - RPMMin')
% ylabel('RPM')










