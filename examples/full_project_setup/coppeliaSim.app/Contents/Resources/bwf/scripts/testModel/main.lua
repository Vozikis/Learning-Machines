simBWF=require('simBWF')
local isCustomizationScript=sim.getScriptAttribute(sim.getScriptAttribute(sim.handle_self,sim.scriptattribute_scripthandle),sim.scriptattribute_scripttype)==sim.scripttype_customization

if false then -- if not sim.isPluginLoaded('Bwf') then
    function sysCall_init()
    end
else
    function sysCall_init()
        sim.writeCustomStringData(sim.getObject('..'),'',nil) -- remove all tags and data
        sim.writeCustomStringData(sim.getObject('..'),simBWF.modelTags.TESTMODEL,sim.packTable({version=1})) -- append the tag with data that just contains the version number
        model={}
        simBWF.appendCommonModelData(model,simBWF.modelTags.TESTMODEL)
        if isCustomizationScript then
            -- Customization script
            if model.modelVersion==1 then
                require("/bwf/scripts/testModel/common")
                require("/bwf/scripts/testModel/customization_data")
                require("/bwf/scripts/testModel/customization_main")
                require("/bwf/scripts/testModel/customization_dlg")
                require("/bwf/scripts/testModel/customization_ext")
                --[[
                --]]
            end
        end
        sysCall_init() -- one of above's 'require' redefined that function
    end
end
