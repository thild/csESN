using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

namespace csESN
{
    class Program
    {
        static void Main(string[] args)
        {
            Mackey.Run();
            // var summary = BenchmarkRunner.Run<ESNBenchmark>();

        }
    }

    [MemoryDiagnoser]
    [SimpleJob(RunStrategy.Monitoring, launchCount: 1, warmupCount: 0, targetCount: 1)]
    public class ESNBenchmark
    {
        public ESNBenchmark()
        {
        }

        [Benchmark]
        public void Run()
        {
            Mackey.Run();
        }
    }
}
