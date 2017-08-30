require 'nn'
require 'rnn'
require 'os'

local batch_size = 5
local feat_dim = 6
local hidden_size = 4
local seq_len = 10
local num = 2 
local lr = 0.01
--------initialize model
--[[
local model = nn.SeqLSTM(feat_dim,hidden_size)
--local model = nn.Sequencer(nn.Linear(feat_dim,hidden_size))
model:clearState()
torch.save("model_init_seqLSTM.t7", model)
os.exit()
--]]
local model1 = torch.load("model_init_seqLSTM.t7")
--local params,  gradparams =  model:getParameters()
--local criterion = nn.SequencerCriterion(nn.MSECriterion())
-------------input, label
local input = {}
local gradOut = {}
for i = 1, num do
	x = torch.randn(seq_len,batch_size,feat_dim)
	y = torch.randn(seq_len,batch_size,hidden_size)
	table.insert(input,x)
	table.insert(gradOut,y)
end
-------------------new way
local models,out,gradInputs = {},{},{}
for i = num,1,-1 do
	local model = torch.load("model_init_seqLSTM.t7")
	models[i] = model
	if i < num then
		params_cur, gradParams_cur = models[i]:getParameters()
		params_updated, gradParams_updated = models[i+1]:getParameters()
		for j = 1, (#params_cur)[1] do
			params_cur[j] = params_updated[j]
			gradParams_cur[j] = gradParams_updated[j]
		end
	end
	out[i] = models[i]:forward(input[i])
	gradInputs[i] = models[i]:backward(input[i],gradOut[i])
	models[i]:updateParameters(0.01)
	models[i]:forget()
	models[i]:zeroGradParameters()
end
params_updated, gradParams_updated = models[1]:getParameters()
for i = 2,num do
	params_cur, gradParams_cur = models[i]:getParameters()
        for j = 1, (#params_cur)[1] do
	        params_cur[j] = params_updated[j]
                gradParams_cur[j] = gradParams_updated[j]
        end
end
----------------------true one
loss = 0
out={}
for i = num,1,-1 do
        out = model1:forward(input[i])
        gradInputs = model1:backward(input[i], gradOut[i])
	model1:updateParameters(lr)
	model1:forget()
	model1:zeroGradParameters()
end
----------check results
for i = 1,num do
	out0 = models[i]:forward(input[i])
	out1 = model1:forward(input[i])

	print(i,out0,out1)
end

