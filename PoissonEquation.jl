### Loading packages
using Lux, NeuralPDE, ModelingToolkit, Optimization, OptimizationOptimJL
using ModelingToolkit: Interval


### Defining parameters
@variables x y
@parameters u(..)
@derivatives Dxx'' ~ x
@derivatives Dyy'' ~ y

### 2D PDEs
eq = Dxx(u(x, y)) + Dyy(u(x, y)) ~ -(pi * x) * (pi * y)


### boundary conditions
bcs=[(u(0,y)~0.0),
    (u(1,y)~0.0),
    (u(x,0)~0.0),
    (u(x,1)~0.0)]



#### Time span or Domain
domain = [
    x ∈ Interval(0.0, 1.0),
    y ∈ Interval(0.0, 1.0)
]


#### Neural Network
dim=2   #### number of dimesnions coz we have 2 inputs (x and y)
chain=Lux.Chain(Lux.Dense(dim,16, Lux.σ),Lux.Dense(16, 16,Lux.σ), Lux.Dense(16,1))  ## Vector containing d dim input and 1 d output
###  Sigmoid Function, we input | 2 dimesions, 3 layers (16 neurons),sigmoid=activation function,1 output coz we have only 1 equation |


#### discretization
dx=0.1
discretization = NeuralPDE.PhysicsInformedNN(chain, NeuralPDE.GridTraining(dx)) ## GridTraining is Training strategies
# No need to write loss function separately. this module PhysicsInformedNN will automaticaly do this for us.


#### PDE System
@named pdesys = PDESystem(eq, bcs, domain, [x, y], [u(x, y)])
prob = NeuralPDE.discretize(pdesys, discretization)


### Optimizers (We are trying to optimize the weight w of NN by using Gradient descent procedures so that LOSS is Minimised)
opt=OptimizationOptimJL.BFGS()

callback=function(p,l)
    println("current loss is: $l")
    return false 
end

res=Optimization.solve(prob, opt,callback=callback,maxiters=1000)  #### 1000 iterations
phi=discretization.phi  #### returns our minimised solution


#### Plots
xs,ys= [infimum(d.domain):(dx/10):supremum(d.domain) for d in domain]

u_predict=reshape([first(phi([x,y],res.u)) for x in xs for y in ys],
                  (length(xs),length(ys)))


using Plots
p2=plot(xs, ys, u_predict,linetype=:contourf!,fill=true, title="predict")
plot(p2)