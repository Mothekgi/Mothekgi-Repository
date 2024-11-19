using QuantizedArrays

v = collect(1:10);

qv = QuantizedArray(v, k=3)

m = reshape(collect(1:60), (6,10))

qm = QuantizedArray(m, k = 5, m = 2)


using LinLogQuantization
using Test

@testset "minops" begin
    @test minpos([0,0,0,0]) == 0
    @test minpos([0,0,1,0]) == 1

    A = randn(100)
    @test minpos(A) > 0
    @test minpos(A) == minimum([A.>0])

    A = rand(100) .+ 1
    @test minpos(A) == minimum(A)
end

@testset "Linear Quantisation" begin
    for T in [Float64,Float32, Float16]
        for s in [(100,),
                        (10,20),
                        (13,14,15),
                        (23,17,12,5)]

            A = rand(T,s...)

            for LinQ in [LinQuant8Array,
                        LinQuant16Array,
                        LinQuant24Array,
                        LinQuant32Array]

                A2 = Array{T}(LinQ(A))

                @test A2 == Array{T}(LinQ(A2))
            end
        end
    end
end


@testset "Log Quantisation" begin
    for T in [Float64,Float32,Float16]
        for s in [(100,),
                    (10,20),
                    (13,14,15),
                    (23,17,12,5)]

                A = rand(T,s...)

                for LogQ in [LogQuant8Array,
                            LogQuant16Array,
                            LogQuant24Array,
                            LogQuant32Array]

                    for rn in [:linspace,:logspace]

                        A2 = Array{T}(LogQ(A,rn))

                        @test A2 == Array{T}(LogQ(A2,rn))
                    
                end
            end
        end
    end
end

                    
                
@testset "LinQuant along dimension" begin
    A = rand(Float32,10,20,30,40)
    Q = LinQuant32Array(A,4)
    @test A = Array{Float32}(Q)

    Q = LinQuant24Array(A,4)
    @test A = Array{Float32}(Q)

    Q = LinQuant16Array(A,4)
    @test A = Array{Float32}(Q)

    Q = LinQuant8Array(A,4)
    @test all(isapprox.(A,Array{Float32}(Q),atol=1e-1))
end

@testset "LogQuant along dimension" begin
    A = rand(Float32,10,20,30,40)
    Q = LogQuant32Array(A,4)
    @test A = Array{Float32}(Q)

    Q = LogQuant24Array(A,4)
    @test A = Array{Float32}(Q)

    Q = LogQuant16Array(A,4)
    @test A = Array{Float32}

    Q = LogQuant8Array(A,4)
    @test all(isapprox.(A,Array{Float32}(Q),atol=1e-1))
end







#