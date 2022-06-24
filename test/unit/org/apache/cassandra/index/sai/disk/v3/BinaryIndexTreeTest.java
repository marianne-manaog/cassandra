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

package org.apache.cassandra.index.sai.disk.v3;

import org.junit.Test;

import org.agrona.collections.IntArrayList;
import org.apache.cassandra.db.marshal.Int32Type;
import org.apache.cassandra.index.sai.IndexContext;
import org.apache.cassandra.index.sai.SAITester;
import org.apache.cassandra.index.sai.disk.format.IndexDescriptor;
import org.apache.cassandra.index.sai.disk.v1.SegmentMetadata;
import org.apache.cassandra.index.sai.utils.SaiRandomizedTest;

import static org.apache.cassandra.index.sai.disk.v3.BlockTermsTest.toBytes;

public class BinaryIndexTreeTest extends SaiRandomizedTest
{
    @Test
    public void testSimple() throws Exception
    {
        final IndexDescriptor indexDescriptor = newIndexDescriptor();
        final String index = newIndex();
        final IndexContext indexContext = SAITester.createIndexContext(index, Int32Type.instance);

        final BlockTerms.Writer writer = new BlockTerms.Writer(1024, indexDescriptor, indexContext, false);

        final int count = 60_000;

        int value = 0;

        int rowCount = 0;

        long rowid = 0;
        for (int i = 0; i < count; i++)
        {
            int valueCount = nextInt(1, 50);
            for (int x = 0; x < valueCount; x++)
            {
                writer.add(toBytes(value), rowid++);
                rowCount++;
            }
            value++;
        }

        SegmentMetadata.ComponentMetadataMap components = new SegmentMetadata.ComponentMetadataMap();
        writer.finish(components);

        try (V3PerIndexFiles indexFiles = new V3PerIndexFiles(indexDescriptor, indexContext, false);
             BlockTerms.Reader reader = new BlockTerms.Reader(indexDescriptor,
                                                              indexContext,
                                                              indexFiles,
                                                              components))
        {
            BinaryTreeUpperPostings.Writer upperWriter = new BinaryTreeUpperPostings.Writer(reader);

            upperWriter.finish(null, false);

//            for (int x = 0; x < 5; x++)
//            {
//                int min = nextInt(0, value);
//                int max = nextInt(min, value);
//
//                PostingList postings1 = reader.search(toBytes(min), toBytes(max));
//                LongArrayList array1 = collect(postings1);
//
//                PostingList postings2 = reader.searchLeaves(toBytes(min), toBytes(max));
//                LongArrayList array2 = collect(postings2);
//
//                assertArrayEquals(array2.toLongArray(), array1.toLongArray());
//            }
        }
    }

    public void test2() throws Exception
    {


//        UpperPostings.MetaCRC finish(final SegmentMetadata.ComponentMetadataMap components,
//        final boolean segmented) throws IOException
    }

    @Test
    public void test()
    {
        BinaryTreeIndex.BKDPointTree tree = new BinaryTreeIndex.BKDPointTree(10, 10200);

        IntArrayList pathToRoot = new IntArrayList();
        tree.traverse(pathToRoot);


    }
}
