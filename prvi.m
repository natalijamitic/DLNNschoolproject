clc, clear, close all

N = 1000;
A = 2;
B = 5;
f1 = 5;
f2 = 6;

ulaz = linspace(0,6,N);
f = A*sin(2*pi*f1*ulaz) + B*sin(2*pi*f2*ulaz);

izlaz = f + randn(1,N)*0.2*min(A,B);

figure("Name", "Fja bez suma i sa")
hold all
plot(ulaz, f, 'r')
plot(ulaz, izlaz, 'b')

ind = randperm(N);

ulazTrening = ulaz(:, ind(1:0.8*N));
izlazTrening = izlaz(:,ind(1:0.8*N));
ulazTest = ulaz(:, ind(0.8*N+1:N));
izlazTest = izlaz(:,ind(0.8*N+1:N));

net = fitnet([10 10 15 20]);

net.performFcn= 'mse';
net.divideFcn='';
net.trainParam.epochs= 500;
net.trainParam.goal= 0.000001;

[net, tr] = train(net,ulazTrening,izlazTrening);
fPred = sim(net,ulaz);

figure
plotperform(tr)

figure
plotregression(izlazTrening, net(ulazTrening), "Regression")

figure("Name", "Fja sa sumom i predikcija iste"), hold all
plot(ulaz, f, 'b', ulaz, fPred, 'r');