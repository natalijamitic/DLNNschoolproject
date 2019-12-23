clc, clear, close all

load 'dataset2.mat'

pod = rot90(data,3);
ulaz = pod(1:2,:);
izlaz = pod(3,:);

K1 = ulaz(:,izlaz==0); 
K2 = ulaz(:,izlaz==1);

figure("Name", "Vizualizovanje podataka po klasama"), hold all
scatter(K1(1,:), K1(2,:),'r.');
scatter(K2(1,:), K2(2,:),'b.');

N = length(ulaz);
ind = randperm(N);

ulazTrening = ulaz(:, ind(1:0.8*N));
izlazTrening = izlaz(:,ind(1:0.8*N));

ulazTest = ulaz(:, ind(0.8*N+1:N));
izlazTest = izlaz(:,ind(0.8*N+1:N));

net1 = patternnet([50 50 50 100 200]);
net2 = patternnet([20 1]); %under
net3 = patternnet([50 50 100 100 100 200 200 200]); %over

net1.trainParam.epochs= 500;
net2.trainParam.epochs= 500;
net3.trainParam.epochs= 500;

net1.trainParam.goal = 0.000001;
net2.trainParam.goal = 0.000001;
net3.trainParam.goal = 0.000001;

net1.divideFcn='';
net2.divideFcn='';
net3.divideFcn='';

[net1, tr1] = train(net1,ulazTrening,izlazTrening);
[net2, tr2] = train(net2,ulazTrening,izlazTrening);
[net3, tr3] = train(net3,ulazTrening,izlazTrening);

figure
plotperform(tr1)
figure
plotperform(tr2)
figure
plotperform(tr3)

izlazPredTest1 = sim(net1, ulazTest);
izlazPredTest2 = sim(net2, ulazTest);
izlazPredTest3 = sim(net3, ulazTest);

figure
plotconfusion(izlazTest, izlazPredTest1);
title('Confusion Matrix 1 - TEST', 'Fontsize',14)
figure
plotconfusion(izlazTest, izlazPredTest2);
title('Confusion Matrix 2 - TEST', 'Fontsize',14)
figure
plotconfusion(izlazTest, izlazPredTest3);
title('Confusion Matrix 3 - TEST', 'Fontsize',14)

izlazPredTrening1 = sim(net1,ulazTrening);
izlazPredTrening2 = sim(net2,ulazTrening);
izlazPredTrening3 = sim(net3,ulazTrening);

figure 
plotconfusion(izlazTrening, izlazPredTrening1);
title('Confusion Matrix 1 - TRAIN', 'Fontsize',14)
figure
plotconfusion(izlazTrening, izlazPredTrening2);
title('Confusion Matrix 2 - TRAIN', 'Fontsize',14)
figure
plotconfusion(izlazTrening, izlazPredTrening3);
title('Confusion Matrix 3 - TRAIN', 'Fontsize',14)

Ntest = 200;
xTest = linspace(-10, 10, Ntest);
yTest = linspace(-10, 10, Ntest);
ulazTestGO = [];
for i = xTest
    ulazTestGO = [ulazTestGO [i*ones(size(yTest)); yTest]];
end

izlazTest1 = sim(net1,ulazTestGO);
izlazTest2 = sim(net2,ulazTestGO);
izlazTest3 = sim(net3,ulazTestGO);

K1p1 =  ulazTestGO(:,izlazTest1<0.3);
K2p1 = ulazTestGO(:,izlazTest1>0.7); 
Kn1 = ulazTestGO(:,izlazTest1>0.3 & izlazTest1<0.7 );
 
K1p2 =  ulazTestGO(:,izlazTest2<0.3);
K2p2 = ulazTestGO(:,izlazTest2>0.7); 
Kn2 = ulazTestGO(:,izlazTest2>0.3 & izlazTest2<0.7 );
 
K1p3 =  ulazTestGO(:,izlazTest3<0.3);
K2p3 = ulazTestGO(:,izlazTest3>0.7); 
Kn3 = ulazTestGO(:,izlazTest3>0.3 & izlazTest3<0.7 );

figure("Name", "Prvi net")
plot(K1p1(1,:),K1p1(2,:),'r.',K2p1(1,:),K2p1(2,:),'b.', Kn1(1,:),Kn1(2,:), 'g.');
 
figure("Name", "Drugi net")
plot(K1p2(1,:),K1p2(2,:),'r.',K2p2(1,:),K2p2(2,:),'b.', Kn2(1,:),Kn2(2,:), 'g.');
 
figure("Name", "Treci net")
plot(K1p3(1,:),K1p3(2,:),'r.',K2p3(1,:),K2p3(2,:),'b.', Kn3(1,:),Kn3(2,:), 'g.');