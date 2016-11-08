require('nngraph')
require('cunn')
require('nnx')

GDL, gdlparent = torch.class('nn.GDLCriterion', 'nn.Criterion')
function GDL:__init(alpha)
    gdlparent:__init(self)
    self.alpha = alpha or 1
    --assert(alpha == 2) --for now
    local Y = nn.Identity()()
    local Yhat = nn.Identity()()
    local Yi1 = nn.SpatialZeroPadding(0,0,0,-1)(Y)
    local Yj1 = nn.SpatialZeroPadding(0,0,-1,0)(Y)
    local Yi2 = nn.SpatialZeroPadding(0,-1,0,0)(Y)
    local Yj2 = nn.SpatialZeroPadding(-1,0,0,0)(Y)
    local Yhati1 = nn.SpatialZeroPadding(0,0,0,-1)(Yhat)
    local Yhatj1 = nn.SpatialZeroPadding(0,0,-1,0)(Yhat)
    local Yhati2 = nn.SpatialZeroPadding(0,-1,0,0)(Yhat)
    local Yhatj2 = nn.SpatialZeroPadding(-1,0,0,0)(Yhat)
    local term1 = nn.Abs()(nn.CSubTable(){Yi2, Yi1})
    local term2 = nn.Abs()(nn.CSubTable(){Yhati2,  Yhati1})
    local term3 = nn.Abs()(nn.CSubTable(){Yj2, Yj1})
    local term4 = nn.Abs()(nn.CSubTable(){Yhatj2, Yhatj1})
    local term12 = nn.CSubTable(){term1, term2}
    local term34 = nn.CSubTable(){term3, term4}
    self.net = nn.gModule({Yhat, Y}, {term12, term34})
    self.net:cuda()
    self.crit = nn.ParallelCriterion()
    if alpha == 2 then
        self.crit:add(nn.MSECriterion())
        self.crit:add(nn.MSECriterion())
    else
        self.crit:add(nn.AbsCriterion())
        self.crit:add(nn.AbsCriterion())
    end
    self.crit:cuda()
    self.target1 = torch.CudaTensor()
    self.target2 = torch.CudaTensor()
end

function GDL:updateOutput(input, target)
    self.netoutput = self.net:updateOutput{input, target}
    self.target1:resizeAs(self.netoutput[1]):zero()
    self.target2:resizeAs(self.netoutput[2]):zero()
    self.target = {self.target1, self.target2}
    self.loss = self.crit:updateOutput(self.netoutput, self.target)
    return self.loss
end

function GDL:updateGradInput(input, target)
    local gradInput =
        self.crit:updateGradInput(self.netoutput, self.target)
    self.gradInput =
        self.net:updateGradInput({input, target}, gradInput)[1]
    return self.gradInput
end
