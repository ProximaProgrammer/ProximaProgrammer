#assuming a simple cylindrical geometry for now
#might have to change 5500 to 5000 in index.html line w/ const response = await fetch('http://127.0.0.1:5500/api', {
#using Render to host website

import math
import torch
import torch.nn as NN
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS # Helps allow the HTML to talk to Python

app = Flask(__name__)
CORS(app) # This prevents "Cross-Origin" errors


thruster_exp = []
with open("thrusterdata.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        thruster_exp.append(row)
   
print(thruster_exp)


# #default/typical values
# xi = None #Ramsauer-Townsend correction constant
# L = 0.05 #length along cathode tube
# R = 0.10 #radius of cathode tube
# r = 0.001 #radius of the central axial anode
# T = 500 #temperature of the gas thrust into cathode tube
# max_power = 265.0 #total power constraint
# A_0 = 2*R*R #cross-sectional area of gas thrust into cathode tube
# mu = 1.5e-4 #ion mobility


GAS_CONFIG = {
    "Argon": -1.5e-3,
    "Krypton": -1.1e-3,
    "Xenon": -0.8e-3,
    "Helium": 0,
    "Neon": 0
}
#old: @app.route('/api', methods=['POST'])
@app.route('/')
def home():
    # This looks inside the 'templates' folder automatically
    return render_template('index.html')

def handle_request():
    data = request.json
    if not data:
        return jsonify({"error": "No data received"}), 400

    xi = None
    gas_key = data.get("gas")
    L = data.get("length")
    max_power = data.get("power")
    T = data.get("temperature")
    r = 0.001*data.get("anode_radius")
    R = r*data.get("radius_ratio")
    A_0 = 2*R*R

    if gas_key in GAS_CONFIG:
        xi = GAS_CONFIG[gas_key]
    else:
        return jsonify({"error": "Invalid gas"}), 400
    if not (isinstance(max_power, (int, float)) and max_power > 0):
        return jsonify({"error": "Invalid power value"}), 400


    print(f"Received: {gas_key}, Power: {max_power}, Radius: {R}")

    if L <= 0:
        return jsonify({"error": "Length must be > 0"}), 400

    if r <= 0:
        return jsonify({"error": "Anode radius must be > 0"}), 400

    # Return actual values so the alert shows something real
    total_thrust = main_function(xi, L, max_power, T, r, R, A_0)
    return jsonify({
        "thrust": str(total_thrust/1e6) + " N (" + str(total_thrust) + " µN)",
        "status": "Success",
        "gas_used": gas_key
    })

def main_function(xi, L, max_power, T, r, R, A_0):
    # check that setting L_custom = L is physically sound

    # output: V0(x) at boundary (so replace t with x)        inputs: cat hode-anode electrode gap d, cylinder radius R
    #want to maximize: output thrust (and optionally, ionization fraction)
    #NOTE: After PINN can find an adequate model, change x data to linspace and worry about experimental data for its x values (do a discrete sum)?

    N = 20 #number of internal nodes
    #x_min, x_max = 0.0,1.0
    #x_data = np.linspace(x_min, x_max, N)
    x_data = np.array([float(thruster_exp[i][0]) for i in range(1,11)])
    dx_data = np.concatenate([np.zeros(1), np.array([float(thruster_exp[i+1][0]) - float(thruster_exp[i][0]) for i in range(1, 10)])])
    y_data = np.array([float(thruster_exp[i][1]) for i in range(1,11)]) #experimental data’s V(z)


    scaler_y = StandardScaler()
    y_data_scaled = scaler_y.fit_transform(y_data.reshape(-1,1))


    #loss weights
    lambda_data = 1.0
    lambda_physics = 3.0
    lambda_power_constraint = 60.0
    lambda_IC = 1.0


    num_epochs = 4000
    print_every = 1000


    def true_solution(x): #simply return experimental values
        return x_data[np.where(x_data == x)[0][0]]


    class PINN(NN.Module):
        def __init__(self, n_hidden=16):
            super(PINN, self).__init__()
            # A simple MLP with 2 hidden layers
            self.net = NN.Sequential(
                NN.Linear(1, n_hidden),
                NN.Tanh(),
                NN.Linear(n_hidden, n_hidden),
                NN.Tanh(),
                NN.Linear(n_hidden, 1)
            )


        def forward(self, x):
            """
            Forward pass: input shape (batch_size, 1) -> output shape (batch_size, 1)
            """
            return self.net(x)


    def derivative(y,x):
        return torch.autograd.grad(
            y,x,
            grad_outputs = torch.ones_like(y),
            create_graph = True
        )[0]


    def nth_derivative(y,x,n):
        if n==1:
            return derivative(y,x)
        return nth_derivative(derivative(y,x),x,n-1)


    model = PINN(n_hidden = 20)
    #y_data_exact = true_solution(x_data)
        #y_data is the thrust_force_uN column of data


    x_data_tensor = torch.tensor(x_data, dtype=torch.float32).view(-1, 1)
    dx_data_tensor = torch.tensor(dx_data, dtype=torch.float32).view(-1,1)
    y_data_tensor = torch.tensor(y_data_scaled, dtype=torch.float32).view(-1, 1)


    def mask(thrust):
        if thrust<60000:
            return 1e4 * (60000-thrust)
        elif thrust>120000:
            return 1e4 * (thrust-120000)
        return 0


    def physics_loss(model, x, dx):
        x.requires_grad_(True)
        y_scaled = model(x)
        prediction = y_scaled * torch.tensor(scaler_y.scale_, dtype=torch.float32) + torch.tensor(scaler_y.mean_, dtype=torch.float32)


        #prediction = model(x) #V(x) = model(x)
        nothing = 1e-6
        thrust = dx * (1 + x*(R*R/(L*r*r))) * nth_derivative(prediction, x, 2) * derivative(prediction, x) * (1 + xi*derivative(prediction, x) + ((1.67188203e-5/T) * (torch.clamp(torch.abs(nth_derivative(prediction, x, 2)), min=1e-4))**(1/3)))
    #NOTE: FIX THIS
        return mask(torch.sum(thrust)) + torch.mean(-1*torch.tanh(thrust)) #note: added tanh to prevent going hyper on maximizing thrust


    def IC_loss(model):
    #     #modify this (see notebook) | up to proportionality constant, optimal: 130 V for first half of x range, then 130 → 50 for next quarter, then 50 for next quarter
    #     t0 = torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
    #     V0_prediction = model(x0)
    #     return (V0_prediction - y0).pow(2).mean()
    #    return torch.zeros(1, 1, dtype=torch.float32, requires_grad=False)
        x1 = torch.tensor([x_data[0]], dtype=torch.float32).view(-1, 1)
        x1.requires_grad_(True)
        x2 = torch.tensor([x_data[-1]], dtype=torch.float32).view(-1, 1)
        x2.requires_grad_(True)
        V1 = model(x1) * torch.tensor(scaler_y.scale_, dtype=torch.float32) + torch.tensor(scaler_y.mean_, dtype=torch.float32)
        V2 = model(x2) * torch.tensor(scaler_y.scale_, dtype=torch.float32) + torch.tensor(scaler_y.mean_, dtype=torch.float32)
        return (V1/V2)/(4.0)


    def data_loss(model, x_data, y_data):
        y_prediction = model(x_data)
        return torch.mean((y_prediction - y_data)**2) #normalized at each point


    def power_constraint_loss(model, x, dx): #should the parameter be x or x_data? #potential difference depends on plasma, hence the derivatives
        V = model(x)
        # Unscale for physics
        #V = v_scaled * torch.tensor(scaler_y.scale_, dtype=torch.float32) + torch.tensor(scaler_y.mean_, dtype=torch.float32)
    
        dv_dx = derivative(V, x)
        d2v_dx2 = nth_derivative(V, x, 2)
        d2v_dx2_safe = torch.clamp(torch.abs(d2v_dx2), min=1.0)


        P_desired = abs(dx * (1 + x*(R*R/(L*r*r))) * d2v_dx2 * dv_dx * (V + 13.5 + 8.61423221e-5*T*torch.log(d2v_dx2_safe/1e14)))
        P_actual = max_power
        return ((torch.sum(P_desired)-P_actual)/P_actual)**2 #add a torch.sum(P_desired) term --> don't let power be negative


    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005) #lr=0.0015 and 0.0001 works nice
    print("python line 217 successfully reached!")

    model.train()
    for epoch in range(500): #First make model realistic by fitting to experimental data
        optimizer.zero_grad()


        l_data = data_loss(model, x_data_tensor, y_data_tensor)
        loss = lambda_data * l_data
    
        loss.backward()
        optimizer.step()


    for epoch in range(500, num_epochs): #Now adhere to physics and power constraint, which are more important
        optimizer.zero_grad()


        l_physics = physics_loss(model, x_data_tensor, dx_data_tensor)
        l_IC =  IC_loss(model)
        l_power_constraint = power_constraint_loss(model, x_data_tensor, dx_data_tensor)
        loss =  lambda_physics * l_physics + lambda_IC * l_IC + lambda_power_constraint * l_power_constraint


        #Backpropagation
        loss.backward()
        optimizer.step()


        if (epoch+1)%print_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, " f"Total Loss = {loss.item():.6f}, " f"Data Loss = {l_data.item():.6f},  " f"Physics/Thrust Loss = {l_physics.item():.6f}, " f"Power Constraint Loss = {l_power_constraint.item():.6f}, " f"IC Loss = {l_IC.item():.6f}")


    model.eval()


    # Do NOT use torch.no_grad() here because we need derivatives
    x_phys = torch.tensor(x_data, dtype=torch.float32).view(-1, 1).requires_grad_(True)


    # 1. Forward pass
    y_scaled = model(x_phys)


    # 2. Unscale Voltage (Keep tensors for the graph)
    scale = torch.tensor(scaler_y.scale_, dtype=torch.float32)
    mean = torch.tensor(scaler_y.mean_, dtype=torch.float32)
    V_phys = y_scaled * scale + mean


    # 3. Calculate derivatives
    # These will work now because we didn't use no_grad()
    dVdx = derivative(V_phys, x_phys)
    d2Vdx2 = nth_derivative(V_phys, x_phys, 2)


    # 4. Calculate Thrust
    dx_tensor = torch.tensor(dx_data, dtype=torch.float32).view(-1, 1)
    term1 = dx_tensor * (1 + x_phys * (R**2 / (L * r**2)))
    term2 = d2Vdx2 * dVdx
    term3 = (1 + xi * dVdx + ((1.67188203e-5 / T) * (torch.abs(d2Vdx2) + 1e-9)**(1/3)))


    individual_thrusts = term1 * term2 * term3
    total_thrust = torch.sum(individual_thrusts).item()
    return total_thrust

# For live server based testing during coding:
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

# if power is proportional to V^n, just let everything except length be initially chosen, then use a dimensional scaling argument to apply L to handle exp. Data??
# x_scaling_factor = L_custom/thruster_exp[10][0]
# Total_thrust *= x_scaling_factor**

# print(f"Total Model Thrust: {total_thrust:.4f} μN")


### Save the optimized weights/biases ###
# torch.save(pinn_model.state_dict(), 'pinn_state.pt')
# # transfer save into a new model
# new_pinn = PINN()
# new_pinn.load_state_dict(torch.load('pinn_state.pt'))
# new_pinn.train() # Continue training or fine-tuning


# x_plot = np.linspace(x_min, x_max, 100).reshape(-1, 1).astype(np.float32)
# x_plot_tensor = torch.tensor(x_plot, requires_grad = True)
# y_prediction_plot = model(x_plot_tensor).detach().numpy()


# # For plotting only: use np.interp to handle values between data points
# y_true = np.interp(x_plot.flatten(), x_data.flatten(), y_data.flatten())
# plt.figure(figsize=(8,5))
# plt.scatter(x_data, y_data, color='red', label = 'Experimental Data')
# plt.plot(x_plot, y_true, 'k--', label = 'Exact Solution')
# plt.plot(x_plot, y_prediction_plot, 'b', label = 'PINN Prediction')
# plt.xlabel('t')
# plt.ylabel('y')
# # plt.xlim(-0.1,0.1)
# # plt.ylim(-1e-5,1e-5)
# plt.legend()
# plt.title('PINN for Thrust')
# plt.grid(True)
# plt.show()
