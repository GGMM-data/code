clear
figure(1)

energy_consumption_ratio = 0.25:0.5:2.75;

%centrilized_average_score = [0.5961	0.7735461	0.7913928	0.7654405	0.7817884	0.673102];
%centrilized_average_score = [0.5961	0.7735461	0.7913928	0.7654405	0.7817884	0.72699];
centrilized_average_score = [0.735	0.7735461	0.7913928	0.7654405	0.7817884	0.74699];
mtsp_average_score = [0.7904 0.7904 0.7904 0.7904 0.7904 0.7904];
distributed_average_score = [0.846693	0.8495322	0.9203726	0.878381	0.905488	0.8707784];
greedy_average_score=[0.794501
	0.7984656
	0.804134
	0.7806045999999999
	0.8094722
	0.8140874

];
ramdom_average_score=[0.7331068
	0.7515832
	0.7309844
    0.7445887999999999
		0.7644612
	0.7610124

];
plot(energy_consumption_ratio,distributed_average_score,'r-*',energy_consumption_ratio,mtsp_average_score ,'m-.^',energy_consumption_ratio,centrilized_average_score,'g->',energy_consumption_ratio,greedy_average_score,'b-o',energy_consumption_ratio,ramdom_average_score,'k-s','Linewidth',2.5,'markersize',10)

xlim([0,3])
ylim([0.7,1.05])

set(gca,'fontsize',20)
set(gca,'xtick',(0.25:0.5:2.75),'ytick', (0.7:0.1:1.05))
set(gca,'xticklabel',{'0.2','0.3','0.4','0.5','0.6','0.7'})

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel({'Normalized energy consumption' ,'for flying with max distance (unit)'},'fontsize',20)
ylabel('Average coverage score','fontsize',20)
grid on;
saveas(gcf,'ratio_score.pdf')
