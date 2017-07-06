/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.cassandra.tools.nodetool;

import static com.google.common.base.Preconditions.checkArgument;
import static org.apache.commons.lang3.StringUtils.join;
import io.airlift.airline.Arguments;
import io.airlift.airline.Command;
import io.airlift.airline.Option;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import javax.management.openmbean.CompositeData;
import javax.management.openmbean.OpenDataException;
import javax.management.openmbean.TabularDataSupport;

import org.apache.cassandra.db.ColumnFamilyStoreMBean;
import org.apache.cassandra.metrics.TableMetrics;
import org.apache.cassandra.metrics.TableMetrics.Sampler;
import org.apache.cassandra.tools.NodeProbe;
import org.apache.cassandra.tools.NodeTool.NodeToolCmd;

import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;

@Command(name = "toppartitions", description = "Sample and print the most active partitions for a given column family")
public class TopPartitions extends NodeToolCmd
{
    @Arguments(usage = "<keyspace> <cfname> <duration>", description = "The keyspace, column family name, and duration in milliseconds")
    private List<String> args = new ArrayList<>();
    @Option(name = "-s", description = "Capacity of stream summary, closer to the actual cardinality of partitions will yield more accurate results (Default: 256)")
    private int size = 256;
    @Option(name = "-k", description = "Number of the top partitions to list (Default: 10)")
    private int topCount = 10;
    @Option(name = "-a", description = "Comma separated list of samplers to use (Default: all)")
    private String samplers = join(TableMetrics.Sampler.values(), ',');
    @Override
    public void execute(NodeProbe probe)
    {
        checkArgument(topCount < size, "TopK count (-k) option must be smaller then the summary capacity (-s)");

        if (args.size() == 1)
        {
            topParittionsForAllTables(probe);
            return;
        }

        checkArgument(args.size() == 3, "toppartitions requires keyspace, column family name, and duration");
        String keyspace = args.get(0);
        String cfname = args.get(1);
        Integer duration = Integer.valueOf(args.get(2));
        ColumnFamilyStoreMBean cfProxy = probe.getCfsProxy(keyspace, cfname);
        System.out.print(topPartitions(probe, cfProxy, keyspace, cfname, duration));
    }

    private void topParittionsForAllTables(NodeProbe probe)
    {
        Integer duration = Integer.valueOf(args.get(0));

        ExecutorService executorService = Executors.newCachedThreadPool();
        try
        {
            List<Future<String>> futures = new ArrayList<>();
            for (Iterator<Entry<String, ColumnFamilyStoreMBean>> cfProxyIter = probe.getColumnFamilyStoreMBeanProxies(); cfProxyIter.hasNext(); )
            {
                Entry<String, ColumnFamilyStoreMBean> entry = cfProxyIter.next();
                String ks = entry.getKey();
                ColumnFamilyStoreMBean cfProxy = entry.getValue();
                String table = cfProxy.getTableName();

                futures.add(executorService.submit(() -> topPartitions(probe, cfProxy, ks, table, duration)));
            }

            for (Future<String> f : futures)
            {
                try
                {
                    System.out.print(f.get());
                }
                catch (InterruptedException e)
                {
                    Thread.currentThread().interrupt();
                    break;
                }
                catch (ExecutionException e)
                {
                    throw new RuntimeException(e.getCause());
                }
            }
        }
        finally
        {
            executorService.shutdown();
        }
    }

    private String topPartitions(NodeProbe probe, ColumnFamilyStoreMBean cfProxy, String keyspace, String cfname, Integer duration)
    {
        // generate the list of samplers
        List<Sampler> targets = Lists.newArrayList();
        for (String s : samplers.split(","))
        {
            try
            {
                targets.add(Sampler.valueOf(s.toUpperCase()));
            } catch (Exception e)
            {
                throw new IllegalArgumentException(s + " is not a valid sampler, choose one of: " + join(Sampler.values(), ", "));
            }
        }

        Map<Sampler, CompositeData> results;
        try
        {
            results = probe.getPartitionSample(keyspace, cfname, cfProxy, size, duration, topCount, targets);
        } catch (OpenDataException e)
        {
            throw new RuntimeException(e);
        }
        boolean first = true;
        StringWriter sw = new StringWriter();
        try (PrintWriter pw =  new PrintWriter(sw))
        {
            pw.println("\nKeyspace/table: " + keyspace + '/' + cfname);
            for (Entry<Sampler, CompositeData> result : results.entrySet())
            {
                CompositeData sampling = result.getValue();
                // weird casting for http://bugs.sun.com/view_bug.do?bug_id=6548436
                List<CompositeData> topk = (List<CompositeData>) (Object) Lists.newArrayList(((TabularDataSupport) sampling.get("partitions")).values());
                topk.sort(new Ordering<CompositeData>()
                {
                    public int compare(CompositeData left, CompositeData right)
                    {
                        return Long.compare((long) right.get("count"), (long) left.get("count"));
                    }
                });
                if (!first)
                    pw.println();
                pw.println(result.getKey().toString() + " Sampler:");
                pw.printf("  Cardinality: ~%d (%d capacity)%n", sampling.get("cardinality"), size);
                pw.printf("  Top %d partitions:%n", topCount);
                if (topk.size() == 0)
                {
                    pw.println("\tNothing recorded during sampling period...");
                }
                else
                {
                    int offset = 0;
                    for (CompositeData entry : topk)
                        offset = Math.max(offset, entry.get("string").toString().length());
                    pw.printf("\t%-" + offset + "s%10s%10s%n", "Partition", "Count", "+/-");
                    for (CompositeData entry : topk)
                        pw.printf("\t%-" + offset + "s%10d%10d%n", entry.get("string").toString(), entry.get("count"), entry.get("error"));
                }
                first = false;
            }
        }
        return sw.toString();
    }
}
