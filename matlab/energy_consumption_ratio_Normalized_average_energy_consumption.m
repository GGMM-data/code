clear
figure(1)

energy_consumption_ratio = 0.25:0.5:2.75;
%centrilized_Normalized_average_energy_consumption=[0.2	0.306853522083	0.407274841723	0.507330639454	0.604066195397	0.703604695546];
centrilized_Normalized_average_energy_consumption=[0.22	0.306853522083	0.407274841723	0.507330639454	0.604066195397	0.703604695546];
mtsp_Normalized_average_energy_consumption = [1 1 1 1 1 1];
distributed_Normalized_average_energy_consumption = [0.290795563235	
    0.367330041815	
    0.4518858086146488	
    0.563452	
    0.64737	
    0.734590760237];
greedy_Normalized_average_energy_consumption=[0.28
	0.370629370629
	0.461077844311
	0.5500000000000052
	0.64
	0.72972972973
];
ramdom_Normalized_average_energy_consumption=[0.261246521878
	0.354232404513
	0.447011699038
	0.538270131490073
	0.630583756194
	0.722683733592
];

plot(energy_consumption_ratio,distributed_Normalized_average_energy_consumption,'r-*',energy_consumption_ratio,mtsp_Normalized_average_energy_consumption ,'m-.^',energy_consumption_ratio,centrilized_Normalized_average_energy_consumption,'g->',energy_consumption_ratio,greedy_Normalized_average_energy_consumption,'b-o',energy_consumption_ratio,ramdom_Normalized_average_energy_consumption,'k-s','Linewidth',2.5,'markersize',10)

xlim([0,3])
ylim([0.15,1.15])

set(gca,'fontsize',20)
set(gca,'xtick',(0.25:0.5:2.75),'ytick', (0.2:0.1:1.15))
set(gca,'xticklabel',{'0.2','0.3','0.4','0.5','0.6','0.7'})

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel({'Normalized energy consumption' ,'for flying with max distance (unit)'},'fontsize',20)
ylabel({'Average energy consumption'},'fontsize',20)
%ylabel('Normalized Average Energy Consumption','fontsize',20)
grid on;
saveas(gcf,'ratio_energy.pdf')
