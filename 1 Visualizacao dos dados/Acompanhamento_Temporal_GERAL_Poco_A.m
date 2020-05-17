

INTERVALOX = [118 124];
figure
% subplot(7,1,1)
% plot(LAS_RM_TIME_HOUR, LAS_RM_TIME_COBTM, 'k.')
% legend('LAS - BitOnBotton')
% ylabel('m')
% ylim([-0.2 1.2])
% grid on
% xlim(INTERVALOX)

subplot(5,1,1)
plot(LAS_RM_TIME_HOUR, LAS_RM_TIME_RPM, 'b.')%,...
%      LAS_RM_TIME_HOUR, LAS_RM_TIME_CRPM, 'r.')
%legend('LAS - RPM','LAS - Collar RPM')
ylabel('RPM')
ylim([0, 250])
grid on
xlim(INTERVALOX)

subplot(5,1,2)
plot(LAS_RM_TIME_HOUR, LAS_RM_TIME_STOR, 'b.')
%legend('LAS - STOR')
ylabel('1000 ft.lbf ')
ylim([0, 60])
grid on
xlim(INTERVALOX)

subplot(5,1,3)
plot(LAS_RM_TIME_HOUR, LAS_RM_TIME_SWOB, 'b.')
%legend('LAS - SWOB')
ylabel('1000 lbf')
ylim([0, 100])
grid on
xlim(INTERVALOX)

subplot(5,1,4)
plot(LAS_RM_TIME_HOUR, LAS_RM_TIME_DEPTH, 'b.')
%legend('LAS - DEPTH')
ylabel('DEPTH (m)')
ylim([4950 5025])
grid on
xlim(INTERVALOX)

% subplot(7,1,6)
% plot(R5K_ElapsedTime+1.425-0.005, R5K_CentripetalAcc, 'r.',...
%     BURST_PLUG_ElapsedTime+1.425-0.005, BURST_PLUG_AngularSpeed, 'k.')
% legend('R5K - Accel Cent.', 'BURST PLUG - RPM')
% ylabel('g')
% grid on
% xlim(INTERVALOX)

subplot(5,1,5)
plot(R6K_ElapsedTime-0.75+0.011, R6K_RPMMax, '.',...
     R6K_ElapsedTime-0.75+0.011, R6K_RPMMean, '.',...
     R6K_ElapsedTime-0.75+0.011, R6K_RPMMin, '.')
%     BURST_ElapsedTime_M-0.75+0.011, BURST_RPM_M, 'r.')
xlim(INTERVALOX)
ylim([0, 500])
grid on
%legend('R6K(BBHD) - RPMMax', 'R6K(BBHD) - RPMMean', 'R6K(BBHD) - RPMMin', 'BURST HD - RPM')
ylabel('RPM')
xlabel('Elapsed Time (h)')





%%
ini = 60
fim = 62
% figure()
%%
ini = ini+2
fim = fim+2
INTERVALOX = [ini fim];
%%

% INTERVALOX = [65.957 65.961];  BURST_PLUG_ElapsedTime_sync = BURST_PLUG_ElapsedTime*1.285 + 2 -0.13;
INTERVALOX = [65.957 65.961];

%%


subplot(3,1,1)
plot(R5K_ElapsedTime_sync, R5K_CentripetalAcc, 'b-')
legend('R5K CentripetalAcc')   
ylabel('g')
grid on
xlim(INTERVALOX)


BURST_PLUG_ElapsedTime_sync = BURST_PLUG_ElapsedTime(395839:end-6);
% BURST_PLUG_ElapsedTime_sync = BURST_PLUG_ElapsedTime_sync*1.28;
BURST_PLUG_ElapsedTime_sync = BURST_PLUG_ElapsedTime_sync*1.285;
BURST_PLUG_ElapsedTime_sync = BURST_PLUG_ElapsedTime_sync + 2 -0.105;

subplot(3,1,2)
plot(BURST_PLUG_ElapsedTime_sync, BURST_PLUG_Acceleration(395839:end-6), 'b-')
legend('Burst Plug Accel')   
ylabel('g')
grid on
xlim(INTERVALOX)

% subplot(4,1,3)
% plot(R5K_ElapsedTime_sync, R5K_MaxLateralAcc, 'b-')
% legend('R5K MaxLateralAcc')   
% xlim(INTERVALOX)
% ylabel('g')
% grid on

subplot(3,1,3)
plot(R6K_ElapsedTime-0.75+0.011, R6K_RPMMax*(2*pi/60),...
     R6K_ElapsedTime-0.75+0.011, R6K_RPMMean*(2*pi/60),...
     R6K_ElapsedTime-0.75+0.011, R6K_RPMMin*(2*pi/60),...
     BURST_ElapsedTime_M-0.75+0.011, BURST_RPM_M*(2*pi/60), 'r.')
xlim(INTERVALOX)
ylim([0, 500*(2*pi/60)])
grid on
legend('R6K(BBHD) - RPMMax', 'R6K(BBHD) - RPMMean', 'R6K(BBHD) - RPMMin', 'BURST HD - RPM')
ylabel('rad/s')




%%

figure()

INTERVALOX = [124.35 124.45];

subplot(4,1,3)
plot(R5K_ElapsedTime_sync, R5K_MaxLateralAcc, 'b-')
legend('R5K MaxLateralAcc')   
xlim(INTERVALOX)
ylabel('g')
grid on

subplot(4,1,4)
plot(R6K_ElapsedTime-0.75+0.011, R6K_RPMMax*(2*pi/60),...
     R6K_ElapsedTime-0.75+0.011, R6K_RPMMean*(2*pi/60),...
     R6K_ElapsedTime-0.75+0.011, R6K_RPMMin*(2*pi/60),...
     BURST_ElapsedTime_M-0.75+0.011, BURST_RPM_M*(2*pi/60), 'r.')
xlim(INTERVALOX)
ylim([0, 500*(2*pi/60)])
grid on
legend('R6K(BBHD) - RPMMax', 'R6K(BBHD) - RPMMean', 'R6K(BBHD) - RPMMin', 'BURST HD - RPM')
ylabel('rad/s')



