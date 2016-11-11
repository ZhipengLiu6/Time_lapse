-- adapted from: wojciechz/learning_to_execute on github
require 'nn'
require 'nngraph'
require 'torch'
-- Creates one timestep of one LSTM
--nngraph.setDebug(true)
function mylstm(inputsize, outputsize) ---
    local x = -nn.Identity()
    local prev_c = -nn.Identity()
    local prev_h = -nn.Identity()

    function new_input_sum()
        -- transforms input
        local i2h            = x - nn.Linear(inputsize, outputsize)
        -- transforms previous timestep's output
        local h2h            = prev_h - nn.Linear(inputsize, outputsize)
        return nn.CAddTable()({i2h, h2h})
    end

    local in_gate          = nn.Sigmoid()(new_input_sum())
    local forget_gate      = nn.Sigmoid()(new_input_sum())
    local out_gate         = nn.Sigmoid()(new_input_sum())
    local in_transform     = nn.Tanh()(new_input_sum())

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return nn.gModule({x, prev_c, prev_h}, {next_c, next_h})
end
--[[test
inputsize, outputsize = 100, 100
x = torch.randn(100)
prev_c = torch.randn(100)
prev_h = torch.randn(100)
lstm = mylstm(inputsize, outputsize)
output = lstm:forward({x , prev_c, prev_h})
print(output)
os.execute('open -a  Safari my_bad_lstm.svg')
]]--