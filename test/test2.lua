require 'nn'
require 'rnn'
require 'os'

local batch_size = 5
local feat_dim = 6
local hidden_size = 4
local seq_len = 10
local num = 2 
--------initialize model
--[[
local model = nn.SeqLSTM(feat_dim,hidden_size)
--local model = nn.Sequencer(nn.Linear(feat_dim,hidden_size))
model:clearState()
torch.save("model_init_seqLSTM.t7", model)
os.exit()
--]]
local model1 = torch.load("model_init_seqLSTM.t7")
local model2 = torch.load("model_init_seqLSTM.t7")
local model3 = torch.load("model_init_seqLSTM.t7")

--local params,  gradparams =  model:getParameters()

local criterion = nn.SequencerCriterion(nn.MSECriterion())
-------------input, label
local input = {}
local gradOut = {}
for i = 1, num do
	x = torch.randn(seq_len,batch_size,feat_dim)
	y = torch.randn(seq_len,batch_size,hidden_size)
	table.insert(input,x)
	table.insert(gradOut,y)
end
-------way 1
loss = 0
out={}
out=model1:forward(input[num])
for i = num,1,-1 do
        gradInputs = model1:backward(input[i], gradOut[i])
model1:updateParameters(0.01)
--model1:forget()
--model1:zeroGradParameters()
end
-------way 2
loss = 0
out={}
for i = 1,num do
        out[i]=model2:forward(input[i])
end
for i = num,1,-1 do
        gradInputs = model2:backward(input[i], gradOut[i])
model2:updateParameters(0.01)
--model2:forget()
--model2:zeroGradParameters()
end
-------way 3
loss = 0
out={}
for i = 1,num do
        out[i]=model3:forward(input[i])
end
for i = num,1,-1 do
        gradInputs = model3:backward(input[i], gradOut[i])
model3:updateParameters(0.01)
--model3:forget()
--model3:zeroGradParameters()
end
----------check results
for i = 1,num do
	out1 = model1:forward(input[i])
	out2 = model2:forward(input[i])
	out3 = model3:forward(input[i])

	print(i,out1,out2)
end

