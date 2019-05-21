clear
figure(1)

uav_numbers = 3:1:8;

centrilized_average_score = [0.5505418	0.6439577	0.7279919	0.7654405	0.8257958	0.8300623];
mtsp_average_score = [0.5182 0.6388 0.7323 0.7904 0.8451 0.8771];
distributed_average_score = [0.6248606,0.707523,0.762821,0.878381,0.874560,	0.9010326];
greedy_average_score=[ 0.6102232
	0.6965471999999999
	0.7538560000000003
	0.7806045999999999
	0.8308147999999999
    0.884142 
];
ramdom_average_score=[ 0.5086092
	0.6169946
	0.6989214
	0.745469
	0.8015562000000002
	0.8366826

];
plot(uav_numbers,distributed_average_score,'r-*',uav_numbers, mtsp_average_score,'m-.^',uav_numbers,centrilized_average_score,'g->',uav_numbers,greedy_average_score,'b-o',uav_numbers,ramdom_average_score,'k-s','Linewidth',2.5,'markersize',10)

xlim([2.7,8.3])
ylim([0.45,1])

set(gca,'xtick', (3:1:8),'fontsize',20)
set(gca,'ytick', (0.5:0.1:1),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel('Number of UAVs','fontsize',20)
ylabel('Average coverage score','fontsize',20)
grid on;
saveas(gcf,'UAV_score.pdf')
