### Import packages
using Lux, Optimization, OptimizationOptimJL, NeuralPDE, ModelingToolkit, Plots
using ModelingToolkit: Interval


### defining the variables and parameters
@variables x t
@parameters u(..)
Dt=Differential(t) ## d/dt
Dxx=Differential(x)^2  ## d/dx^2
Dtt=Differential(t)^2  ## d/dt^2


### defining the equation
c=1
eq=Dtt(u(x,t))~ (c^2)*Dxx(u(x,t))


### boundary conditions
bcs=[u(0,t)~0.0,
    u(1,t)~0.0,
    u(x,0)~x*(1-x),
    Dt(u(x,0))~0.0]


### time span
domian=[x ∈ Interval(0.0,1.0),
        t ∈ Interval(0.0,1.0)]

#### Forming a Neural Network using sigmoid function
dim=2
chain=Lux.Chain(Lux.Dense(dim,16,Lux.σ),Lux.Dense(16,16, Lux.σ),Lux.Dense(16,1))


##### discretization
dx=0.1
discretization=NeuralPDE.PhysicsInformedNN(chain,NeuralPDE.GridTraining(dx))

#### PDE System
@named PDEsys=PDESystem(eq, bcs, domian, [x,t],[u(x,t)])
prob=NeuralPDE.discretize(PDEsys,discretization)



##### Optimization
opt=OptimizationOptimJL.BFGS()

callback=function (p,l)
    println("Current loss is: $l")
    return false    
end


res=Optimization.solve(prob,opt,callback=callback,maxiters=(800))
phi=discretization.phi


#### Plotting the precdicted values
xs,ts=[infimum(d.domain):dx:supremum(d.domain) for d in domian]
u_predict=reshape([first(phi([x,t],res.u)) for x in xs for t in ts],
                    (length(xs),length(ts)))


function analytic_sol_func(t, x) 
    sum([(8 / (n^3 * pi^3)) * sin(n * pi * x) * cos(c * n * pi * t) for n in 1:2:10000])
end

u_analytic=reshape([analytic_sol_func(t, x) for x in xs for t in ts],
                    (length(xs),length(ts)))


p1=plot(xs,ts, u_predict,linetype=:contourf!,fill=true, title="Predicted Solution")
p2=plot(xs,ts, u_analytic,linetype=:contourf!,fill=true, title="Analytical/True Solution")
plot(p1,p2)