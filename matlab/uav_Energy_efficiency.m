clear
figure(1)

uav_numbers = 3:1:8;

centrilized_Energy_efficiency = [0.654571606312	0.892796811842	1.13362628163	1.23169859449	1.41461328992	1.42986606046];
mtsp_Energy_efficiency = [0.4686 0.5987 0.7072 0.7698 0.8300 0.8658];
distributed_Energy_efficiency = [0.8346916311871857	
    1.0057558251600274	
    1.1709711571013817	
    1.435410	
    1.472280881474347	
    1.54231145];
greedy_Energy_efficiency=[0.7084379764342674
	0.9220036719325635
	1.066975621234361
	1.1460801566620855
	1.2836274578520908
	1.4177822343300917

];
ramdom_Energy_efficiency=[0.5034359495722397
	0.7407099438927718
	0.9367213627358333
	1.062542410815768
	1.2219460228578274
	1.326396701361981

];
plot(uav_numbers,distributed_Energy_efficiency,'r-*',uav_numbers,mtsp_Energy_efficiency ,'m-.^',uav_numbers,centrilized_Energy_efficiency,'g->',uav_numbers,greedy_Energy_efficiency,'b-o',uav_numbers,ramdom_Energy_efficiency,'k-s','Linewidth',2.5,'markersize',10)

xlim([2.7,8.3])
ylim([0.4,1.7])

set(gca,'xtick', (3:1:8),'fontsize',20)
set(gca,'ytick', (0.4:0.2:1.7),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel('Number of UAVs','fontsize',20)
ylabel('Energy efficiency','fontsize',20)
grid on;
saveas(gcf,'UAV_effi.pdf')
