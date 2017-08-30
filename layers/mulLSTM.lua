require 'rnn'
require 'nn'
--require 'cutorch'
--require 'cunn'
local mulLSTM, Parent = torch.class('nn.mulLSTM', 'nn.Module')

function mulLSTM:__init(opt)
	Parent.__init(self)
        self.feat_dim = utils.getopt(opt, 'feat_dim')
        self.hidden_size = utils.getopt(opt, 'hidden_size')
        self.seq_length = utils.getopt(opt, 'shot_length')
        self.event_num =  utils.getopt(opt, 'event_num')
        self.batch_size = utils.getopt(opt, 'batch_size')

	--[[
	self.model = nn.Sequencer(
	nn.Sequential()
		:add(nn.FastLSTM(self.feat_dim, self.hidden_size):maskZero(1))
--		:add(nn.MaskZero(nn.Linear(hiddenSize, numTargetClasses),1))
--		:add(nn.MaskZero(nn.LogSoftMax(),1))
	)
	--]]
	self.model = nn.SeqLSTM(self.feat_dim, self.hidden_size)
	self.model.maskzero = true
end

function mulLSTM:parameters()
        local p1,g1 = self.model:parameters()

        local params = {}
        for k,v in pairs(p1) do table.insert(params, v) end

        local grad_params = {}
        for k,v in pairs(g1) do table.insert(grad_params, v) end

        return params, grad_params
end

function mulLSTM:updateOutput(input)
	-- input: seqlen x batchsize x feat_dim
	-- output: seqlen x batchsize x hidden_size
	if input:type() == 'torch.CudaTensor' then
                --d_hidden_ = d_hidden_:cuda()
                self.model:cuda()
        end
	local output = self.model:forward(input)	

	--only need the output of last timestep
	return output[self.seq_length]
end

function mulLSTM:updateGradInput(input, gradOutput)
	local gradOutput_all = torch.Tensor(self.seq_length, self.batch_size, self.hidden_size):fill(0)
	gradOutput_all[self.seq_length] = gradOutput
	
	self.gradInput = self.model:backward(input,gradOutput_all)
	return self.gradInput
end
