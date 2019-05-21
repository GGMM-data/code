clear
figure(1)

energy_consumption_ratio = 0.25:0.5:2.75;


%centrilized_Energy_efficiency = [1.666235	2.03877943205	1.62160485941	1.23169859449	1.05901384729	0.676593378135];
%centrilized_Energy_efficiency = [1.666235	2.03877943205	1.62160485941	1.23169859449	1.05901384729	0.8094238084357];
centrilized_Energy_efficiency = [2.52	2.03877943205	1.62160485941	1.23169859449	1.05901384729	0.8094238084357];
mtsp_Energy_efficiency = [0.7698 0.7698 0.7698 0.7698 0.7698 0.7698];
distributed_Energy_efficiency = [2.6795183267	
    2.08995673697	
    1.925333602890587	
    1.435410	
    1.33001257762
    1.10497145398];
greedy_Energy_efficiency=[2.32481889874
	1.77294820168
	1.44337485979
	1.1460801566620855
	1.05470309586
	0.935303752509

];
ramdom_Energy_efficiency=[2.12210360262
	1.64190079669
	1.23454536634
	1.0596660836098
	0.9514965623
	0.824869062074
];
plot(energy_consumption_ratio,distributed_Energy_efficiency,'r-*',energy_consumption_ratio,mtsp_Energy_efficiency ,'m-.^',energy_consumption_ratio,centrilized_Energy_efficiency,'g->',energy_consumption_ratio,greedy_Energy_efficiency,'b-o',energy_consumption_ratio,ramdom_Energy_efficiency,'k-s','Linewidth',2.5,'markersize',10)

xlim([0,3])
ylim([0.5,3])

set(gca,'fontsize',20)
set(gca,'xtick',(0.25:0.5:2.75),'ytick', (0.5:0.5:3))
set(gca,'xticklabel',{'0.2','0.3','0.4','0.5','0.6','0.7'})

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthEast','fontsize',13)

xlabel({'Normalized energy consumption' ,'for flying with max distance (unit)'},'fontsize',20)
ylabel('Energy efficiency','fontsize',20)
grid on;
saveas(gcf,'ratio_effi.pdf')
