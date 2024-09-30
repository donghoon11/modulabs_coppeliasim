sim=require'sim'
simVision=require'simVision'

function sysCall_init()
    visionSensorHandles={}
    for i=1,4,1 do
        visionSensorHandles[i]=sim.getObject('../sensor',{index=i-1})
    end
    ptCloudHandle=sim.getObject('../ptCloud')
    frequency=5 -- 5 Hz
    options=2+8 -- bit0 (1)=do not display points, bit1 (2)=display only current points, bit2 (4)=returned data is polar (otherwise Cartesian), bit3 (8)=displayed points are emissive
    pointSize=2
    coloring_closeAndFarDistance={1,4}
    displayScaling=0.999 -- so that points do not appear to disappear in objects
    h=simVision.createVelodyneVPL16(visionSensorHandles,frequency,options,pointSize,coloring_closeAndFarDistance,displayScaling,ptCloudHandle)
end

function sysCall_sensing()
    -- h + sim.hanleflag_abscoords : velodyne 센서의 데이터를 처리할 때 절대 좌표계를 사용하라.
    data=simVision.handleVelodyneVPL16(h+sim.handleflag_abscoords,sim.getSimulationTimeStep())

    -- if we want to display the detected points ourselves:
    if ptCloud then
        sim.removePointsFromPointCloud(ptCloud,0,nil,0)
    else
        ptCloud=sim.createPointCloud(0.02,20,0,pointSize)
    end
    sim.insertPointsIntoPointCloud(ptCloud,0,data)

end


function sysCall_cleanup()
    simVision.destroyVelodyneVPL16(h)
end