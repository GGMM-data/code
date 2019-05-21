clear
figure(1)

uav_numbers = 3:1:8;

centrilized_Normalized_average_energy_consumption=[0.504268961468	0.509436443883	0.507774840509	0.507330639454	0.504839019451	0.51105080963];
mtsp_Normalized_average_energy_consumption = [1 1 1 1 1 1];
distributed_Normalized_average_energy_consumption = [0.5321640691172577	
    0.5300850641038677	
    0.5307780957407516	
    0.563452	
    0.5479846582099743	
    0.5470721];
greedy_Normalized_average_energy_consumption=[0.5500000000000052
	0.5500000000000052
	0.5500000000000052
	0.5500000000000052
	0.5500000000000052
	0.5500000000000052

];
ramdom_Normalized_average_energy_consumption=[0.5383437904428185
	0.5382785354725043
	0.53823981173122
	0.538282181037578
	0.5382516460077271
	0.5382499044017804

];

plot(uav_numbers,distributed_Normalized_average_energy_consumption,'r-*',uav_numbers,mtsp_Normalized_average_energy_consumption ,'m-.^',uav_numbers,centrilized_Normalized_average_energy_consumption,'g->',uav_numbers,greedy_Normalized_average_energy_consumption,'b-o',uav_numbers,ramdom_Normalized_average_energy_consumption,'k-s','Linewidth',2.5,'markersize',10)

xlim([2.7,8.3])
%ylim([0.48,0.68])
ylim([0.45,1.05])

set(gca,'xtick', (3:1:8),'fontsize',20)
%set(gca,'ytick', (0.48:0.04:0.68),'fontsize',20)
set(gca,'ytick', (0.45:0.1:1.05),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel('Number of UAVs','fontsize',20)
ylabel({'Average energy consumption'},'fontsize',20)
%ylabel('Normalized average energy consumption','fontsize',20)
grid on;
saveas(gcf,'UAV_energy.pdf')
